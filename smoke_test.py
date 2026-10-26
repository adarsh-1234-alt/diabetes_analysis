"""Smoke test for the Diabetes Risk Predictor app.

Runs the real Streamlit script in-process (all eight clinician tabs execute on
every run) and checks input scenarios plus the Stage 2, 3, 4 and 5 features:
threshold tuning, the uncertainty interval, the four-model comparison, the
matplotlib gauge, colour-coded status pills, the before/after comparison,
plain-language explanation cards, range bars, the try-change sentence, the
recommendation engine, patient comparison, saved history, batch scoring, the
PDF report and the patient/clinician view split.

Usage:
    python smoke_test.py

Exit code 0 = all checks passed, 1 = something failed.
"""
import importlib.util
import json
import os
import sys
import warnings

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402

import report_utils  # noqa: E402

APP = "app.py"
METRICS = os.path.join(os.path.dirname(os.path.abspath(APP)), "metrics.json")
N_INPUTS = 8

# Sidebar widget order (see app.py):
#   number_input[0] = Pregnancies
#   slider[0] Glucose | [1] BloodPressure | [2] SkinThickness | [3] Insulin
#   slider[4] BMI     | [5] DPF           | [6] Age


def page_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [i.value for i in at.info] + [w.value for w in at.warning]
    parts += [s.value for s in at.success] + [e.value for e in at.error]
    parts += [s.value for s in at.subheader]
    return "\n".join(parts)


def report(at, tag):
    print(f"\n=== {tag} ===")
    print("  exceptions:", [f"{e.type}: {e.value}" for e in at.exception])
    print("  errors    :", [e.value[:110] for e in at.error])
    print("  metrics   :", [(m.label, m.value) for m in at.metric])
    print("  dataframes:", len(at.dataframe), "| tabs:", len(at.tabs),
          "| plotly:", len(at.get("plotly_chart")))
    return not at.exception


def check_stage2(at, tag):
    """Stage 2 assertions that must hold in every run."""
    text = page_text(at)
    checks = {
        "threshold shown":   "Screening threshold" in text,
        "interval shown":    "90% interval" in text,
        "threshold section": "Decision threshold tuning" in text,
        "xgboost listed":    "XGBoost" in text,
        "smote removed":     "SMOTE oversampling" not in text,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        print(f"  STAGE2 FAIL: {bad}")
    return not bad


def check_stage3(at, tag):
    """Stage 3 assertions: gauge, status pills, before/after panel."""
    text = page_text(at)
    pills = text.count('class="pill"')
    has_status_col = any(
        "Status" in list(getattr(frame.value, "columns", []))
        for frame in at.dataframe
    )
    checks = {
        "gauge rendered":      len(at.get("imgs")) >= 1,
        "no plotly needed":    len(at.get("plotly_chart")) == 0,
        "status pills":        pills == N_INPUTS,
        "before/after panel":  any("Before / After" in s.value for s in at.subheader),
        "summary has status":  has_status_col,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        print(f"  STAGE3 FAIL: {bad} (pills={pills})")
    return not bad


def check_stage4(at, tag):
    """Stage 4 assertions: LIME, dependence, interaction, recommendations."""
    text = page_text(at)
    labels = [metric.label for metric in at.metric]
    tab_labels = [tab.label for tab in at.tabs]
    checks = {
        "eight tabs":           len(at.tabs) == 8,
        "history tab":          any("History" in name for name in tab_labels),
        "batch tab":            any("Batch" in name for name in tab_labels),
        "report tab":           any("Report" in name for name in tab_labels),
        "explain headline":    any("Risk is" in s.value for s in at.subheader),
        "raises cards":        "What raises this result" in text,
        "lowers cards":        "What lowers it" in text,
        "range bars":          "Where this patient sits" in text,
        "try change":          "Try a change" in text,
        "recommendation":      ("What to improve first" in text
                                and "Change this first" in labels),
        "comparison toggle":   len(at.toggle) == 1,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        print(f"  STAGE4 FAIL: {bad}")
    return not bad


def check_stage5(at, tag):
    """Stage 5 assertions that must hold in every clinician run."""
    text = page_text(at)
    buttons = [button.label for button in at.button]
    checks = {
        "save control":    any("Save to history" in label for label in buttons),
        "batch section":   "Batch prediction" in text,
        "pdf section":     "PDF report" in text,
        "pdf built":       "Report risk" in [metric.label for metric in at.metric],
        "no pdf error":    not any("PDF" in error.value for error in at.error),
        "privacy note":    "session's memory" in text,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        print(f"  STAGE5 FAIL: {bad}")
    return not bad


def scenario(tag, mutate=None, assertions=None):
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["auth"] = True
    at.run()
    ok = report(at, f"{tag} — defaults")
    ok &= check_stage2(at, tag)
    ok &= check_stage3(at, tag)
    ok &= check_stage4(at, tag)
    ok &= check_stage5(at, tag)
    if mutate:
        mutate(at)
        at.run()
        ok &= report(at, f"{tag} — modified")
        ok &= check_stage2(at, tag)
        ok &= check_stage3(at, tag)
        ok &= check_stage4(at, tag)
        ok &= check_stage5(at, tag)
        if assertions:
            ok &= assertions(at)
    return ok


def zero_labs(at):
    at.slider[0].set_value(0)    # Glucose not measured
    at.slider[3].set_value(0)    # Insulin not measured
    at.slider[4].set_value(0.0)  # BMI not measured


def extremes(at):
    at.number_input[0].set_value(20)
    at.slider[0].set_value(250)
    at.slider[5].set_value(2.5)
    at.slider[6].set_value(90)


def improbable(at):
    at.slider[4].set_value(0.1)  # BMI just above the "not measured" sentinel


def high_glucose(at):
    at.slider[0].set_value(220)


def healthy_glucose(at):
    at.slider[0].set_value(90)


def expects_split_decision(at):
    text = page_text(at)
    return "Positive at the tuned threshold" in text


def expects_neighbour_warning(at):
    text = page_text(at)
    return "not measured" in text


def expects_before_after(at):
    """Default glucose (120) exceeds the healthy target, so a comparison exists."""
    labels = [m.label for m in at.metric]
    return "Current risk" in labels and "After improvement" in labels


def expects_abnormal_flag(at):
    text = page_text(at)
    return "abnormal value(s) entered" in text and "Abnormal" in text


def expects_no_comparison(at):
    """Glucose below target: the panel explains there is nothing to model."""
    text = page_text(at)
    return "already at or below the healthy target" in text


def enable_comparison(at):
    at.toggle[0].set_value(True)


def try_change_slider_check(at):
    """Adjusting the Try-a-change feature selector must re-score cleanly without crashing."""
    at.selectbox[1].set_value("BMI")


def expects_comparison(at):
    labels = [m.label for m in at.metric]
    text = page_text(at)
    return ("Difference (B − A)" in labels
            and len(at.number_input) == 9
            and "Feature-by-feature difference" in text
            and ("points lower" in text or "points higher" in text)
            and len(at.get("imgs")) >= 6)


def expects_try_change_rescored(at):
    text = page_text(at)
    return "Overall Clinical Picture" in text and "moves the estimated risk from" in text


def stage5_flow():
    """Multi-step Stage 5 checks: save → batch → patient view → PDF bytes."""
    results = []
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["auth"] = True
    at.run()

    history_ok = False
    try:
        save = next(button for button in at.button
                    if "Save to history" in button.label)
        save.click()
        at.run()
        text = page_text(at)
        history_ok = (not at.exception
                      and len(at.session_state["history"]) == 1
                      and "Risk over time" in text
                      and len(at.get("imgs")) >= 4)
    except Exception as exc:  # noqa: BLE001
        print("  history flow error:", exc)
    results.append(("history save", history_ok))

    batch_ok = False
    try:
        demo = next(box for box in at.checkbox
                    if "example patients" in box.label)
        demo.set_value(True)
        at.run()
        labels = [metric.label for metric in at.metric]
        batch_ok = (not at.exception
                    and "Patients scored" in labels
                    and any("Risk %" in list(getattr(frame.value, "columns", []))
                            for frame in at.dataframe))
    except Exception as exc:  # noqa: BLE001
        print("  batch flow error:", exc)
    results.append(("batch demo", batch_ok))

    patient_ok = False
    try:
        at.radio[0].set_value("Patient")
        at.run()
        text = page_text(at)
        patient_ok = (not at.exception
                      and len(at.tabs) == 0
                      and "Your result" in text
                      and "SHAP" not in text
                      and "LIME" not in text
                      and len(at.get("imgs")) >= 1
                      and any("Save to my progress" in button.label
                              for button in at.button))
    except Exception as exc:  # noqa: BLE001
        print("  patient flow error:", exc)
    results.append(("patient view", patient_ok))

    pdf_ok = False
    try:
        pdf = report_utils.build_report_pdf(
            label="Smoke Test", risk=0.262, level="Low", threshold=0.30,
            decision="Negative", interval=(0.03, 0.50),
            values=[("Plasma Glucose (mg/dL)", "120", "Normal: <140 mg/dL",
                     "Normal")],
            rec_rows=[{"feature": "Plasma Glucose", "current": 120,
                       "target": 100, "unit": "mg/dL", "new_risk": 0.152}],
            flags=["Blood pressure >= 90 mm Hg - high"],
            drivers=[("Plasma Glucose", 0.012)], history_rows=[])
        pdf_ok = (not report_utils.PDF_OK) or (
            pdf is not None and pdf[:5] == b"%PDF-" and len(pdf) > 2000)
    except Exception as exc:  # noqa: BLE001
        print("  pdf flow error:", exc)
        pdf_ok = False
    results.append(("pdf report", pdf_ok))
    return results


def main():
    with open(METRICS, encoding="utf-8") as handle:
        threshold = float(json.load(handle)["threshold"])

    results = [
        ("normal", scenario("A", None, expects_before_after)),
        ("missing labs", scenario("B", zero_labs, expects_neighbour_warning)),
        ("extreme values", scenario("C", extremes, expects_split_decision)),
        ("implausible value", scenario("D", improbable)),
        ("abnormal input", scenario("E", high_glucose, expects_abnormal_flag)),
        ("healthy input", scenario("F", healthy_glucose, expects_no_comparison)),
        ("comparison mode", scenario("G", enable_comparison, expects_comparison)),
        ("try-change & narrative", scenario("H", try_change_slider_check,
                                            expects_try_change_rescored)),
    ]

    # The threshold in metrics.json must be the one the app uses.
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["auth"] = True
    at.run()
    results.append(("threshold wired", f"{threshold:.2f}" in page_text(at)))

    results.extend(stage5_flow())

    # New explain-tab checks (clinician view, default scenario)
    at_explain = AppTest.from_file(APP, default_timeout=900)
    at_explain.session_state["auth"] = True
    at_explain.run()
    explain_text = page_text(at_explain)
    explain_ok = (
        "Risk is" in explain_text
        and "What raises this result" in explain_text
        and "What lowers it" in explain_text
        and "Where this patient sits" in explain_text
        and "Try a change" in explain_text
        and "How sure is this?" in explain_text
        and "SHAP" not in explain_text
        and "LIME" not in explain_text
    )
    results.append(("explain tab", explain_ok))

    # The login gate must still block unauthenticated users.
    at2 = AppTest.from_file(APP, default_timeout=900)
    at2.run()
    results.append(("login gate", not at2.exception and len(at2.tabs) == 0))

    print("\n--- summary ---")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = [name for name, ok in results if not ok]
    print("\nRESULT:", "PASS" if not failed else f"FAIL ({', '.join(failed)})")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
