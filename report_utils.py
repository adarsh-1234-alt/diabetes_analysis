"""PDF report and trend-chart helpers for the Diabetes Risk Predictor.

This module deliberately contains **no Streamlit imports**: the app imports it
at runtime, and `smoke_test.py` imports it directly to verify that a real PDF
is produced. Every public function degrades gracefully instead of raising.

reportlab is optional. If it is missing the app keeps working and simply says
that PDF export is unavailable (`PDF_OK` is False).
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless backend — this module may be imported outside Streamlit
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

try:  # optional dependency
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (CondPageBreak, Image, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    PDF_OK = True
except Exception:  # noqa: BLE001
    PDF_OK = False

BAND_COLOURS = {"Low": "#2e9e4f", "Moderate": "#e0a223", "High": "#d64545"}
STATUS_COLOURS = {
    "Normal": "#1a7f37",
    "Borderline": "#8a6100",
    "Abnormal": "#b42318",
    "Not measured": "#57606a",
}

# Characters that either do not exist in the PDF core fonts or would render as
# boxes. Everything is mapped to ASCII before it reaches reportlab.
_CHAR_MAP = {
    "\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2011": "-",
    "\u2265": ">=", "\u2264": "<=", "\u2260": "!=",
    "\u00b2": "2", "\u00b3": "3", "\u00b5": "u",
    "\u00b7": "-", "\u00b0": " deg", "\u2192": "->", "\u2190": "<-",
    "\u00d7": "x", "\u2026": "...", "\u00a0": " ",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
}


def pdf_safe(value) -> str:
    """Reduce any string to printable ASCII that core PDF fonts can render."""
    text = "" if value is None else str(value)
    for source, target in _CHAR_MAP.items():
        text = text.replace(source, target)
    text = "".join(ch for ch in text if ord(ch) < 128)
    return " ".join(text.split())  # collapse whitespace left by stripped emoji


def _escaped(value) -> str:
    """Escape for reportlab's mini-HTML, after collapsing to safe ASCII."""
    return (pdf_safe(value).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


# ── Trend chart (shared by the app and the PDF) ─────────────────────────────
def _parse_when(value):
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def history_trend_figure(rows, threshold=0.30, width=7.2, height=2.9,
                         compact=False):
    """Build one matplotlib line chart of saved predictions over time.

    `rows` is a list of dicts with `saved_at`, `risk` (0-1) and `label`.
    Returns a Figure, or None when there is nothing usable to plot.
    """
    points = []
    for row in rows or []:
        when = _parse_when(row.get("saved_at"))
        try:
            risk = float(row.get("risk"))
        except (TypeError, ValueError):
            continue
        if when is None:
            continue
        points.append((when, risk, pdf_safe(row.get("label") or "Unnamed patient")))
    if not points:
        return None
    points.sort(key=lambda item: item[0])

    labels = []
    for _, _, name in points:
        if name not in labels:
            labels.append(name)

    palette = ["#1f6feb", "#d64545", "#2e9e4f", "#e0a223", "#7c3aed",
               "#0891b2", "#b45309", "#be185d"]
    fig, ax = plt.subplots(figsize=(width, height))
    for index, name in enumerate(labels):
        series = [(when, risk) for when, risk, label in points if label == name]
        xs = [item[0] for item in series]
        ys = [item[1] * 100 for item in series]
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                color=palette[index % len(palette)], label=name)
        if len(series) == 1:  # make a single visit visible without a line
            ax.scatter(xs, ys, s=46, color=palette[index % len(palette)], zorder=4)

    ax.axhline(threshold * 100, linestyle="--", linewidth=1.2, color="#111827")
    ax.text(points[0][0], threshold * 100 + 2, f"threshold {threshold:.2f}",
            fontsize=7, color="#111827")
    highest = max([risk * 100 for _, risk, _ in points] + [threshold * 100])
    ax.set_ylim(0, min(100, max(55, highest + 12)))
    ax.set_ylabel("Predicted risk (%)", fontsize=8)
    ax.grid(axis="y", color="#eef1f4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    distinct_times = sorted({when for when, _, _ in points})
    if len(distinct_times) == 1:
        # Several visits can share a timestamp (e.g. saved twice in one minute);
        # a date locator would otherwise repeat the same tick label.
        ax.set_xticks(distinct_times)
        ax.set_xticklabels([distinct_times[0].strftime("%d %b\n%H:%M")], fontsize=7)
    else:
        span = distinct_times[-1] - distinct_times[0]
        if span < timedelta(hours=8):
            tick_format = "%H:%M"
        elif span <= timedelta(days=3):
            tick_format = "%d %b\n%H:%M"
        else:
            tick_format = "%d %b"
        ax.xaxis.set_major_formatter(mdates.DateFormatter(tick_format))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=7))
    if len(labels) > 1:
        ax.legend(fontsize=7, frameon=False, loc="best")
    if not compact:
        fig.tight_layout()
    return fig


def figure_png(fig, dpi=160) -> bytes:
    """Render a Figure to PNG bytes and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


# ── PDF report ──────────────────────────────────────────────────────────────
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(rl_colors.HexColor("#6b7280"))
    canvas.drawString(
        18 * mm, 10 * mm,
        "Diabetes Risk Predictor - educational screening tool, not a diagnosis")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=22, textColor=rl_colors.HexColor("#1f2933"),
            alignment=0, spaceAfter=2),
        "subtitle": ParagraphStyle(
            "ReportSub", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.5, leading=11, textColor=rl_colors.HexColor("#57606a")),
        "section": ParagraphStyle(
            "ReportSection", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=rl_colors.HexColor("#1f2933"),
            spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle(
            "ReportBody", parent=base["Normal"], fontName="Helvetica",
            fontSize=8.6, leading=11.5, textColor=rl_colors.HexColor("#1f2933")),
        "small": ParagraphStyle(
            "ReportSmall", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.4, leading=9.6, textColor=rl_colors.HexColor("#57606a")),
        "risk": ParagraphStyle(
            "ReportRisk", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=25, leading=28, alignment=0),
        "cell": ParagraphStyle(
            "ReportCell", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.8, leading=9.8),
        "cell_bold": ParagraphStyle(
            "ReportCellBold", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.8, leading=9.8),
        "cell_head": ParagraphStyle(
            "ReportCellHead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.6, leading=9.6, textColor=rl_colors.HexColor("#40474e")),
    }


def _table(rows, widths, styles, header=True, tint_rows=None):
    """Build a small grid table from already-escaped strings."""
    data = []
    for r_index, row in enumerate(rows):
        style = styles["cell_head"] if (header and r_index == 0) else styles["cell"]
        data.append([Paragraph(_escaped(cell), style) for cell in row])

    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#d8dde3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.white, rl_colors.HexColor("#f7f9fb")]),
    ]
    if header:
        commands.append(("BACKGROUND", (0, 0), (-1, 0),
                         rl_colors.HexColor("#eef1f4")))
    if tint_rows:
        for row_index, colour in tint_rows:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index),
                             rl_colors.HexColor(colour)))
    table.setStyle(TableStyle(commands))
    return table


def build_report_pdf(*, label, risk, level, threshold, decision, interval=None,
                     values=None, rec_rows=None, combined_risk=None, flags=None,
                     drivers=None, history_rows=None, note="", model_name="",
                     generated_at=None, report_id=None) -> bytes | None:
    """Return a finished PDF as bytes, or None when reportlab is unavailable.

    Parameters are plain data so the function is easy to test:
      values       list of (feature, reading, reference, status)
      rec_rows     list of dicts: feature, current, target, unit, new_risk
      drivers      list of (feature, shap_value) or None
      history_rows list of dicts: saved_at, risk, label
    """
    if not PDF_OK:
        return None

    generated_at = generated_at or datetime.now()
    report_id = report_id or generated_at.strftime("%Y%m%d-%H%M%S")
    styles = _styles()
    flag_hex = BAND_COLOURS.get(level, "#1f2933")

    story = [
        Paragraph(_escaped("DIABETES RISK ASSESSMENT"), styles["title"]),
        Paragraph(_escaped("Screening report from the Diabetes Risk Predictor"),
                  styles["subtitle"]),
        Spacer(1, 7 * mm),
    ]

    meta_rows = [
        ["Patient", label or "Unnamed patient", "Report ID", report_id],
        ["Generated", generated_at.strftime("%d %b %Y, %H:%M"),
         "Screening threshold", f"{threshold:.2f}"],
        ["Model", model_name or "not recorded", "Risk band", level],
    ]
    meta = Table(
        [[Paragraph(_escaped(cell), styles["cell_bold"]
                   if column in (0, 2) else styles["cell"])
          for column, cell in enumerate(row)] for row in meta_rows],
        colWidths=[24 * mm, 63 * mm, 33 * mm, 51 * mm])
    meta.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#e6e9ee")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story += [meta, Spacer(1, 6 * mm)]

    result_cell = [
        Paragraph(_escaped(f"{risk * 100:.1f}%"),
                  ParagraphStyle("RiskValue", parent=styles["risk"],
                                 textColor=rl_colors.HexColor(flag_hex))),
        Paragraph(_escaped(
            f"{level} risk band - screening call: {decision.lower()}"),
                  styles["body"]),
    ]
    if interval:
        result_cell.append(Paragraph(_escaped(
            f"90% interval for comparable patients: "
            f"{interval[0] * 100:.1f}% to {interval[1] * 100:.1f}%."),
            styles["small"]))
    result_box = Table([[result_cell]], colWidths=[173 * mm])
    result_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor("#f6f8fa")),
        ("BOX", (0, 0), (-1, -1), 0.6, rl_colors.HexColor("#dfe3e8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [result_box, Spacer(1, 4 * mm)]

    verdict = (
        f"The model estimates a {risk * 100:.1f}% chance of diabetes. "
        + ("This is at or above the tuned screening threshold, so the result "
           "would be flagged for confirmatory testing (fasting plasma glucose "
           "or HbA1c)." if decision == "Positive" else
           "This is below the tuned screening threshold. It does not rule out "
           "diabetes; repeat screening is advised if risk factors change."))
    if any("imputed" in str(item).lower() or "not measured" in str(item).lower()
           for item in (flags or [])):
        verdict += (" Some measurements were not available and were replaced "
                    "with the training median, so the estimate is less certain.")
    story.append(Paragraph(_escaped(verdict), styles["body"]))

    if values:
        story.append(Paragraph(_escaped("Patient measurements"), styles["section"]))
        rows = [["Measurement", "Reading", "Reference", "Status"]]
        tint_rows = []
        for index, (feature, reading, reference, status) in enumerate(values, start=1):
            rows.append([feature, reading, reference, status])
            colour = STATUS_COLOURS.get(status)
            if colour and status in ("Abnormal", "Borderline"):
                tint_rows.append((index, "#fde8e6" if status == "Abnormal"
                                  else "#fff4d6"))
        story.append(_table(rows, [58 * mm, 25 * mm, 60 * mm, 28 * mm],
                            styles, tint_rows=tint_rows))

    if drivers:
        story.append(Paragraph(_escaped("What is driving this estimate"),
                               styles["section"]))
        for feature, value in drivers:
            direction = "raises" if value > 0 else "lowers"
            story.append(Paragraph(_escaped(
                f"- {feature} {direction} the estimated risk "
                f"(SHAP impact {value:+.3f})."), styles["body"]))
        story.append(Paragraph(_escaped(
            "SHAP values describe how the model reached this number. "
            "They are not causes and not treatment effects."), styles["small"]))

    if rec_rows:
        story.append(Paragraph(_escaped("Recommended next steps"),
                               styles["section"]))
        rows = [["Modifiable factor", "Now", "Healthy target",
                 "Modelled risk after", "Change"]]
        for row in rec_rows:
            reduction = (risk - row["new_risk"]) * 100
            rows.append([
                row["feature"],
                f"{row['current']:g} {row['unit']}".strip(),
                f"{row['target']:g} {row['unit']}".strip(),
                f"{row['new_risk'] * 100:.1f}%",
                f"-{reduction:.1f} pts",
            ])
        story.append(_table(rows, [50 * mm, 30 * mm, 34 * mm, 32 * mm, 25 * mm],
                            styles))
        if combined_risk is not None:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(_escaped(
                f"If every target above is reached together, the model "
                f"estimates {combined_risk * 100:.1f}% "
                f"({(risk - combined_risk) * 100:.1f} points lower). Treat "
                "this as an optimistic ceiling, because the model moves one "
                "value at a time while real risk factors interact."),
                styles["small"]))
    else:
        story.append(Paragraph(_escaped("Recommended next steps"),
                               styles["section"]))
        story.append(Paragraph(_escaped(
            "Every modifiable measurement is already at or below its healthy "
            "target. Maintaining these values is the recommendation."),
            styles["body"]))

    if flags:
        story.append(Paragraph(_escaped("Clinical flags"), styles["section"]))
        for flag in flags:
            story.append(Paragraph(_escaped(f"- {flag}"), styles["body"]))

    trend_rows = [row for row in (history_rows or [])
                  if _parse_when(row.get("saved_at")) is not None]
    if len(trend_rows) >= 2:
        story.append(CondPageBreak(95 * mm))
        story.append(Paragraph(_escaped("Progress over time"), styles["section"]))
        fig = history_trend_figure(trend_rows, threshold=threshold,
                                   width=7.2, height=2.6, compact=True)
        if fig is not None:
            story.append(Image(io.BytesIO(figure_png(fig)),
                               width=173 * mm, height=72 * mm))
        recent = sorted(trend_rows, key=lambda row: _parse_when(row["saved_at"]))[-5:]
        rows = [["Visit", "Predicted risk", "Band"]]
        for row in recent:
            when = _parse_when(row["saved_at"])
            rows.append([when.strftime("%d %b %Y, %H:%M"),
                         f"{float(row['risk']) * 100:.1f}%",
                         str(row.get("level") or "")])
        story.append(_table(rows, [80 * mm, 47 * mm, 46 * mm], styles))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(_escaped(
            "Changes between visits reflect changes in the entered "
            "measurements. They are not a measured treatment effect."),
            styles["small"]))

    if note:
        story.append(Paragraph(_escaped("Clinician note"), styles["section"]))
        story.append(Paragraph(_escaped(note), styles["body"]))

    story += [
        Spacer(1, 7 * mm),
        Paragraph(_escaped(
            "This report is generated by a statistical model trained on the "
            "Pima Indians Diabetes Dataset (adult women of Pima heritage, aged "
            "21+). It is an educational screening aid, not a diagnosis, and it "
            "must not replace clinical judgement or confirmatory laboratory "
            "testing. Discuss any positive result with a qualified clinician."),
            styles["small"]),
    ]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Diabetes Risk Assessment",
        author="Diabetes Risk Predictor",
        subject=f"Screening report for {pdf_safe(label or 'unnamed patient')}",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
