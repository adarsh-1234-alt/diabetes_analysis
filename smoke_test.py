"""Smoke test for the Diabetes Risk Predictor app.

Runs the real Streamlit script in-process (all five tabs execute on every run)
and checks input scenarios plus the Stage 2, 3 and 4 features: threshold tuning,
the uncertainty interval, the four-model comparison, the matplotlib gauge, the
colour-coded status pills, the before/after comparison, LIME vs SHAP agreement,
SHAP dependence plots, the feature interaction map, the recommendation engine
and patient comparison mode.

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
        "threshold section": "Decision Threshold Tuning" in text,
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
    checks = {
        "five tabs":           len(at.tabs) == 5,
        "lime section":        "LIME" in text,
        "dependence section":  "SHAP dependence" in text,
        "interaction section": "Feature interaction" in text,
        "recommendation":      ("What to improve first" in text
                                and "Change this first" in labels),
        "comparison toggle":   len(at.toggle) == 1,
    }
    if importlib.util.find_spec("lime") is not None:
        checks["lime agreement"] = "Importance agreement" in labels
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        print(f"  STAGE4 FAIL: {bad}")
    return not bad


def scenario(tag, mutate=None, assertions=None):
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["auth"] = True
    at.run()
    ok = report(at, f"{tag} — defaults")
    ok &= check_stage2(at, tag)
    ok &= check_stage3(at, tag)
    ok &= check_stage4(at, tag)
    if mutate:
        mutate(at)
        at.run()
        ok &= report(at, f"{tag} — modified")
        ok &= check_stage2(at, tag)
        ok &= check_stage3(at, tag)
        ok &= check_stage4(at, tag)
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


def same_interaction_axes(at):
    """Both interaction selectboxes on Glucose: the guard must explain why."""
    at.selectbox[2].set_value("Glucose")


def expects_comparison(at):
    labels = [m.label for m in at.metric]
    text = page_text(at)
    return ("Difference (B − A)" in labels
            and len(at.number_input) == 9
            and "Feature-by-feature difference" in text
            and ("points lower" in text or "points higher" in text)
            and len(at.get("imgs")) >= 8)


def expects_same_feature_guard(at):
    return "Choose two different features" in page_text(at)


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
        ("interaction guard", scenario("H", same_interaction_axes,
                                       expects_same_feature_guard)),
    ]

    # The threshold in metrics.json must be the one the app uses.
    at = AppTest.from_file(APP, default_timeout=900)
    at.session_state["auth"] = True
    at.run()
    results.append(("threshold wired", f"{threshold:.2f}" in page_text(at)))

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
