"""Diabetes Risk Predictor — Streamlit app.

Stage 1 hardening notes
-----------------------
* Every model / SHAP call is wrapped so a failure degrades gracefully instead of
  crashing the whole page.
* Retraining shows a spinner, is time-limited and reports a readable error.
* Clinical sentinel values (0) are treated as "not measured" and imputed, and the
  sidebar surfaces input-validation warnings for implausible values.
* All file paths are resolved relative to this file, so the app works no matter
  which directory `streamlit run` was launched from.

Stage 2 notes
-------------
* The decision uses the recall-tuned threshold produced by train_model.py
  (instead of a hard-coded 0.50) and shows an empirical uncertainty interval.
* XGBoost + class weighting replaced SMOTE; the app reads the threshold and the
  operating-point metrics straight from metrics.json, so old artefacts trigger a
  retrain rather than a wrong answer.

Stage 3 notes
-------------
* The risk figure is a matplotlib speedometer with the three risk bands and the
  tuned threshold marked. Every chart is matplotlib, so no optional plotting
  package can silently disappear on a deployment.
* Every input carries a colour-coded status pill (normal / borderline /
  abnormal / not measured) derived from clinical reference bands, so an
  out-of-range value is visible at a glance rather than buried in a warning list.
* A "before / after" panel re-scores the patient with one modifiable value moved
  to a healthy target and shows the resulting drop in risk.
* Presentation is grouped into cards (`st.container(border=True)`) with a single
  injected stylesheet, so the page reads as one designed surface.

Stage 4 notes
-------------
* Plain-language explanations of the main drivers, using SHAP as a hidden engine.
* A simple CSS range bar shows where the patient sits relative to normal bands.
* Patient comparison scores a second profile side by side with the sidebar one.
* The recommendation engine ranks every modifiable value by how much risk it
  would remove if brought to its healthy target, plus the combined scenario.

Stage 5 notes
-------------
* Saved predictions build a per-patient history with a risk-over-time chart.
  History lives in Streamlit session state (in memory, per session) and can be
  downloaded as CSV/JSON, so nothing is written to a database or file.
* Batch prediction scores a whole CSV in one pass (capped at 2,000 rows) and
  returns the same table plus a risk distribution and a downloadable result.
* A one-click PDF report (reportlab, optional) packages the risk, measurements,
  drivers, recommendations, flags and saved trend. If reportlab is missing the
  app still works and says so.
* Two explicit views: Patient view hides model internals and speaks in plain
  language; Clinician view exposes explainability, model performance, data explorer,
  batch scoring and comparison.
"""
import hmac
import io
import json
import os
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime

import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")  # headless backend — safe inside Streamlit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Make the sibling module importable no matter how the app was launched
# (`streamlit run`, `python -m streamlit`, or Streamlit's AppTest runner).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import report_utils
from report_utils import PDF_OK, build_report_pdf, history_trend_figure

try:  # optional: only needed if the saved model is an XGBoost pipeline
    import xgboost
    XGBOOST_VERSION = xgboost.__version__
except Exception:  # noqa: BLE001
    XGBOOST_VERSION = None

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
BG_PATH      = os.path.join(BASE_DIR, "background.csv")
DATA_PATH    = os.path.join(BASE_DIR, "diabetes.csv")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_model.py")

TRAIN_TIMEOUT       = 600   # seconds before we give up on training
SHAP_INIT_TIMEOUT   = 90
SHAP_ROW_TIMEOUT    = 60

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🩺",
    layout="wide"
)

FEATURES        = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
INT_FEATURES    = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "Age"]

FEATURE_INFO = {
    "Pregnancies": {
        "label": "Number of Pregnancies",
        "help":  "Total number of times pregnant. More pregnancies are linked to hormonal changes that raise diabetes risk.",
        "unit":  "",
        "normal": "0–5 is typical",
    },
    "Glucose": {
        "label": "Plasma Glucose (mg/dL)",
        "help":  "Blood sugar level measured 2 hours after an oral glucose tolerance test. This is the single strongest predictor of diabetes.",
        "unit":  "mg/dL",
        "normal": "Normal: <140 mg/dL",
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure (mm Hg)",
        "help":  "The lower number in a blood pressure reading. High BP is associated with insulin resistance.",
        "unit":  "mm Hg",
        "normal": "Normal: 60–80 mm Hg",
    },
    "SkinThickness": {
        "label": "Triceps Skin Fold Thickness (mm)",
        "help":  "Measures body fat by pinching skin at the triceps. Higher values suggest more body fat.",
        "unit":  "mm",
        "normal": "Normal: 10–30 mm",
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin (mu U/ml)",
        "help":  "Insulin level in blood 2 hours after glucose test. Abnormal levels indicate poor glucose regulation.",
        "unit":  "mu U/ml",
        "normal": "Normal: 16–166 mu U/ml",
    },
    "BMI": {
        "label": "Body Mass Index (kg/m²)",
        "help":  "Weight divided by height squared. Higher BMI means more body fat, which makes it harder for cells to respond to insulin.",
        "unit":  "kg/m²",
        "normal": "Normal: 18.5–24.9",
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "help":  "A score that estimates diabetes likelihood based on family history. Higher score = stronger genetic risk.",
        "unit":  "",
        "normal": "Lower is better",
    },
    "Age": {
        "label": "Age (years)",
        "help":  "Age in years. Diabetes risk naturally increases with age as insulin sensitivity decreases.",
        "unit":  "years",
        "normal": "Risk increases after 45",
    },
}

SENTINEL_HELP = " Set to **0** if this was not measured — the model will estimate it."


# ── Small utilities ─────────────────────────────────────────────────────────
class TrainingError(RuntimeError):
    """Raised when train_model.py cannot produce usable artefacts."""


def run_with_timeout(fn, timeout, label):
    """Run fn() in a daemon thread and raise TimeoutError after `timeout` seconds."""
    box = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
            box["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"{label} did not finish within {timeout} seconds")
    if "error" in box:
        raise box["error"]
    return box["value"]


def safe(fn, timeout, label):
    """Return (result, error_message). Never raises."""
    try:
        return run_with_timeout(fn, timeout, label), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{label} failed — {type(exc).__name__}: {exc}"


def fmt_value(feature, value):
    if value is None or pd.isna(value):
        return "Not measured"
    return f"{value:.0f}" if feature in INT_FEATURES else f"{value:.1f}"


def run_training():
    """Execute train_model.py and raise TrainingError with a readable message."""
    if not os.path.exists(TRAIN_SCRIPT):
        raise TrainingError(f"`train_model.py` was not found in:\n`{BASE_DIR}`")
    if not os.path.exists(DATA_PATH):
        raise TrainingError(
            f"`diabetes.csv` was not found in:\n`{BASE_DIR}`\n\n"
            "Place the Pima Indians Diabetes dataset next to `app.py` and try again.")

    try:
        # Force UTF-8 both ways: Windows defaults to cp1252 for pipes, which
        # breaks on non-ASCII output from the training script.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", TRAIN_SCRIPT],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=TRAIN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise TrainingError(
            f"Training was still running after {TRAIN_TIMEOUT} seconds, so it was stopped. "
            "Try a smaller hyper-parameter grid.") from exc
    except OSError as exc:
        raise TrainingError(f"Could not start Python for training: {exc}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise TrainingError(
            "`train_model.py` exited with an error:\n\n```\n"
            + "\n".join(tail) + "\n```")


# ── Clinical thresholds for flags (NaN-safe) ────────────────────────────────
def get_flags(vals):
    def num(feature):
        value = vals.get(feature, np.nan)
        return None if value is None or pd.isna(value) else float(value)

    flags = []

    glucose = num("Glucose")
    if glucose is not None:
        if glucose >= 200:
            flags.append("🚨 Glucose ≥ 200 mg/dL — strongly indicates diabetes")
        elif glucose >= 140:
            flags.append("🔴 Glucose ≥ 140 mg/dL — elevated blood sugar (pre-diabetic or diabetic range)")

    bmi = num("BMI")
    if bmi is not None:
        if bmi >= 30:
            flags.append("🔴 BMI ≥ 30 — obese range, significantly raises diabetes risk")
        elif bmi >= 25:
            flags.append("🟡 BMI ≥ 25 — overweight range, moderate risk increase")

    bp = num("BloodPressure")
    if bp is not None and bp >= 90:
        flags.append("🔴 Blood pressure ≥ 90 mm Hg — high, linked to insulin resistance")

    insulin = num("Insulin")
    if insulin is not None and insulin >= 167:
        flags.append("🟡 Insulin level above normal range")

    age = num("Age")
    if age is not None and age >= 45:
        flags.append("🟡 Age ≥ 45 — diabetes risk increases significantly after this age")

    dpf = num("DiabetesPedigreeFunction")
    if dpf is not None and dpf >= 0.5:
        flags.append("🟡 High family history score — genetic risk present")

    return flags


# ── Input validation ────────────────────────────────────────────────────────
def validate_inputs(vals, medians):
    """Return (cleaned_values, issues, missing_features).

    A 0 in a column that is clinically impossible as 0 (Glucose, BP, skin
    thickness, insulin, BMI) is interpreted as "not measured" and replaced with
    NaN so the model's own imputer can handle it.
    """
    cleaned, issues, missing = {}, [], []

    for feature in FEATURES:
        value = float(vals[feature])
        if feature in ZERO_AS_MISSING and value <= 0:
            missing.append(feature)
            cleaned[feature] = np.nan
        else:
            cleaned[feature] = value

    for feature in missing:
        median = medians.get(feature)
        filled = f"{median:.1f}" if median is not None and not pd.isna(median) else "median"
        if feature == "Glucose":
            issues.append((
                "warning",
                f"**Glucose is 0** — recorded as *not measured*. The model will substitute "
                f"the training median ({filled} mg/dL). Glucose is the strongest predictor, "
                "so this prediction is much less reliable than a measured reading."))
        else:
            issues.append((
                "warning",
                f"**{FEATURE_INFO[feature]['label']}** is 0 — recorded as *not measured*. "
                f"The model will substitute the training median ({filled})."))

    if len(missing) >= 3:
        issues.append((
            "warning",
            f"**{len(missing)} of the 5 lab measurements are missing.** "
            "The estimate below is a rough guess built mostly from age, pregnancies and family history."))

    glucose = cleaned["Glucose"]
    if pd.notna(glucose) and glucose < 50:
        issues.append((
            "warning",
            "Glucose below 50 mg/dL is outside the range this model was trained on — "
            "treat the result with extra caution."))

    bmi = cleaned["BMI"]
    if pd.notna(bmi) and bmi > 60:
        issues.append((
            "warning",
            "A BMI above 60 kg/m² is outside the training data — the prediction may be unreliable."))
    if pd.notna(bmi) and bmi < 13:
        issues.append((
            "warning",
            "A BMI below 13 kg/m² is implausibly low — check the height/weight entry. "
            "Use 0 only if BMI was not recorded."))

    insulin = cleaned["Insulin"]
    if pd.notna(insulin) and insulin > 700:
        issues.append((
            "warning",
            "An insulin level above 700 mu U/ml is outside the training data — "
            "please double-check the units."))

    if cleaned["Age"] < 21:
        issues.append((
            "info",
            "The Pima dataset only contains adult women aged 21+, so predictions for "
            "younger patients are less reliable."))

    if cleaned["Pregnancies"] >= 10:
        issues.append((
            "info",
            "Very high pregnancy count — only a handful of similar records exist in the "
            "training data, so this feature's effect is less certain."))

    return cleaned, issues, missing


# ── Detailed Plain-English Clinical Explanations (No Graphs) ─────────────────
def explain_feature_up(feat, val):
    if feat == "Glucose":
        if val >= 200:
            return (
                f"**Reading: {val:.0f} mg/dL (Diabetic Range ≥ 200 mg/dL).** "
                "A 2-hour plasma glucose at or above 200 mg/dL is the strongest single indicator of diabetes. "
                "**Why it happens:** Either the pancreas is no longer secreting enough insulin after a meal, or muscle and liver cells have become resistant to insulin's signal, leaving sugar trapped in the bloodstream. "
                "**Clinical next step:** Confirm with a repeat fasting plasma glucose or HbA1c test."
            )
        if val >= 140:
            return (
                f"**Reading: {val:.0f} mg/dL (Impaired Glucose Tolerance: 140–199 mg/dL).** "
                "Blood sugar remains elevated 2 hours after a glucose load (pre-diabetic range). "
                "**Why it happens:** Early insulin resistance forces the pancreas to work harder to clear glucose, causing a delayed drop in blood sugar after meals. "
                "**Clinical next step:** Early dietary carbohydrate management and physical activity can often reverse this stage."
            )
        return (
            f"**Reading: {val:.0f} mg/dL (Upper-Normal Range).** "
            "While technically below the 140 mg/dL pre-diabetes cutoff, this reading sits in the upper half of the normal distribution and slightly nudges the model's estimate upward when combined with other factors."
        )

    if feat == "BMI":
        if val >= 30:
            return (
                f"**Reading: {val:.1f} kg/m² (Obesity Range ≥ 30 kg/m²).** "
                "**Why it happens:** Excess adipose (fat) tissue—especially visceral fat around abdominal organs—releases free fatty acids and inflammatory signals that block insulin receptors on muscle and liver cells (insulin resistance). "
                "**Clinical next step:** A sustained 5–7% reduction in body weight is proven to markedly restore insulin sensitivity."
            )
        if val >= 25:
            return (
                f"**Reading: {val:.1f} kg/m² (Overweight Range: 25.0–29.9 kg/m²).** "
                "**Why it happens:** Carrying extra weight increases the daily demand on pancreatic beta cells to produce more insulin for the same amount of blood sugar clearance. "
                "**Clinical next step:** Modest weight stabilization and resistance exercise help muscle tissue absorb glucose without needing extra insulin."
            )
        return (
            f"**Reading: {val:.1f} kg/m².** "
            "In combination with the patient's other metabolic readings, BMI is adding a small upward pressure on the risk score."
        )

    if feat == "Age":
        return (
            f"**Reading: {val:.0f} years.** "
            "**Why it happens:** As we age (particularly after 45), pancreatic beta-cell turnover slows down and gradual loss of skeletal muscle mass reduces the body's main storage site for blood glucose. "
            "**Clinical next step:** Age cannot be modified, which makes keeping modifiable labs (Glucose, BMI, Blood Pressure) inside target bands even more important."
        )

    if feat == "Pregnancies":
        return (
            f"**Reading: {val:.0f} {'pregnancy' if val == 1 else 'pregnancies'}.** "
            "**Why it happens:** During pregnancy, placental hormones intentionally create temporary insulin resistance to ensure the fetus receives enough glucose. Multiple pregnancies repeatedly stress pancreatic beta cells and can unmask an underlying predisposition to Type 2 diabetes."
        )

    if feat == "Insulin":
        if val > 166:
            return (
                f"**Reading: {val:.0f} mu U/ml (Elevated / Hyperinsulinemia > 166 mu U/ml).** "
                "**Why it happens:** High 2-hour insulin means the pancreas is over-secreting insulin to compensate for insulin-resistant cells. Even if blood sugar looks borderline, high insulin is an early warning sign of metabolic strain."
            )
        if val < 16:
            return (
                f"**Reading: {val:.0f} mu U/ml (Low Insulin Response < 16 mu U/ml).** "
                "**Why it happens:** A low insulin level 2 hours after a glucose challenge can indicate reduced pancreatic beta-cell secretory capacity—meaning the pancreas isn't releasing enough insulin to clear glucose."
            )
        return (
            f"**Reading: {val:.0f} mu U/ml.** "
            "Relative to the patient's blood glucose level, this insulin reading indicates less efficient glucose-insulin regulation."
        )

    if feat == "BloodPressure":
        return (
            f"**Reading: {val:.0f} mm Hg (Diastolic — Normal: 60–80 mm Hg).** "
            "**Why it happens:** Elevated diastolic blood pressure and insulin resistance share common pathways: chronically high insulin causes the kidneys to retain sodium and stiffens blood vessel walls, so hypertension frequently co-occurs with developing diabetes."
        )

    if feat == "SkinThickness":
        return (
            f"**Reading: {val:.0f} mm (Triceps Skin Fold — Normal: 10–30 mm).** "
            "**Why it happens:** A thicker subcutaneous skin fold reflects higher overall body fat reserves, which correlates with reduced peripheral insulin sensitivity."
        )

    if feat == "DiabetesPedigreeFunction":
        return (
            f"**Reading: {val:.2f} (Family History Score — Higher above 0.50).** "
            "**Why it happens:** Type 2 diabetes has a strong polygenic inheritance pattern. A higher pedigree score means first- or second-degree relatives developed diabetes, passing down genes that affect beta-cell resilience and insulin signaling."
        )

    return f"**{feat}** ({val:.2f}) is contributing to a higher risk estimate."


def explain_feature_down(feat, val):
    if feat == "Glucose":
        return (
            f"**Reading: {val:.0f} mg/dL (Healthy Range < 140 mg/dL).** "
            "**Why it protects:** Two hours after a glucose challenge, blood sugar has returned to a normal baseline. This shows that pancreatic insulin secretion and cellular glucose uptake are working in sync."
        )
    if feat == "BMI":
        return (
            f"**Reading: {val:.1f} kg/m² ({'Healthy Range 18.5–24.9' if val < 25 else 'Moderate Range'}).** "
            "**Why it protects:** Keeping body mass in a lower band minimizes visceral fat inflammation, allowing muscle and liver cells to respond cleanly to normal amounts of insulin."
        )
    if feat == "Age":
        return (
            f"**Reading: {val:.0f} years (Under 45).** "
            "**Why it protects:** Younger age is associated with stronger baseline pancreatic beta-cell reserve and higher skeletal muscle insulin sensitivity."
        )
    if feat == "Pregnancies":
        return (
            f"**Reading: {val:.0f} {'pregnancy' if val == 1 else 'pregnancies'}.** "
            "**Why it protects:** Lower cumulative gestational metabolic stress reduces long-term strain on insulin production."
        )
    if feat == "Insulin":
        return (
            f"**Reading: {val:.0f} mu U/ml (Normal Range: 16–166 mu U/ml).** "
            "**Why it protects:** A normal 2-hour serum insulin level suggests the pancreas is regulating blood sugar without having to over-secrete insulin."
        )
    if feat == "BloodPressure":
        return (
            f"**Reading: {val:.0f} mm Hg (Normal Range: 60–80 mm Hg).** "
            "**Why it protects:** Healthy vascular tone is associated with better capillary blood flow to skeletal muscle, supporting healthy glucose uptake."
        )
    if feat == "SkinThickness":
        return (
            f"**Reading: {val:.0f} mm (Normal Range: 10–30 mm).** "
            "**Why it protects:** Subcutaneous fat thickness is within typical bounds and is not adding metabolic resistance."
        )
    if feat == "DiabetesPedigreeFunction":
        return (
            f"**Reading: {val:.2f} (Low Family History Score < 0.50).** "
            "**Why it protects:** A lower pedigree score indicates minimal known hereditary burden among close relatives."
        )
    return f"**{feat}** ({val:.2f}) is helping keep the estimated risk lower."


def explain_missing(feat, median):
    filled = f"{median:.1f}" if median is not None and not pd.isna(median) else "the median"
    return (
        f"**Not measured (Model substituted dataset median: {filled}).** "
        f"Because **{FEATURE_INFO[feat]['label']}** was entered as 0, the model could not use a personal reading. "
        "Measuring this lab directly will make the explanation and score much more specific to this patient."
    )


def synthesize_clinical_narrative(cleaned, risk):
    """Build a 2-3 sentence holistic clinical summary connecting the patient's values."""
    g = cleaned.get("Glucose")
    bmi = cleaned.get("BMI")
    ins = cleaned.get("Insulin")
    bp = cleaned.get("BloodPressure")
    age = cleaned.get("Age")
    dpf = cleaned.get("DiabetesPedigreeFunction")

    mechanisms = []
    if pd.notna(g) and g >= 140:
        mechanisms.append(
            f"elevated 2-hour blood glucose (**{g:.0f} mg/dL**)"
            + (" in the diabetic range" if g >= 200 else " in the impaired-tolerance range")
        )
    if pd.notna(bmi) and bmi >= 25:
        mechanisms.append(
            f"excess body weight (**BMI {bmi:.1f} kg/m²**)"
            + (", which drives cellular insulin resistance" if bmi >= 30 else "")
        )
    if pd.notna(ins) and ins > 166:
        mechanisms.append(f"compensatory high insulin (**{ins:.0f} mu U/ml**)")
    if pd.notna(bp) and bp >= 90:
        mechanisms.append(f"elevated diastolic blood pressure (**{bp:.0f} mm Hg**)")
    if pd.notna(dpf) and dpf >= 0.5:
        mechanisms.append(f"strong family history (**pedigree score {dpf:.2f}**)")
    if pd.notna(age) and age >= 45:
        mechanisms.append(f"age-related beta-cell decline (**{age:.0f} years**)")

    if risk >= THRESHOLD:
        if mechanisms:
            joined = ", ".join(mechanisms[:-1]) + (f" and {mechanisms[-1]}" if len(mechanisms) > 1 else mechanisms[0])
            return (
                f"**Overall Clinical Picture:** This patient's elevated risk (**{risk:.1%}**) is primarily driven by "
                f"{joined}. Together, these factors point toward impaired glucose clearance and increased insulin resistance, "
                "which is why the profile crosses the screening threshold for confirmatory testing (HbA1c or fasting plasma glucose)."
            )
        return (
            f"**Overall Clinical Picture:** Even without a single extreme lab value, the combined pattern of measurements "
            f"places this profile at **{risk:.1%}** risk—above the **{THRESHOLD:.2f}** screening cutoff—warranting a confirmatory blood test."
        )
    else:
        if mechanisms:
            return (
                f"**Overall Clinical Picture:** Overall risk remains below the screening cutoff (**{risk:.1%}**), largely because "
                "core glucose regulation is still compensating well. However, watch points are present ("
                + ", ".join(mechanisms)
                + ") that could push risk higher over time if left unaddressed."
            )
        return (
            f"**Overall Clinical Picture:** This profile sits at a low estimated risk (**{risk:.1%}**). "
            "Blood sugar clearance, body mass, and blood pressure work together in a healthy range with no signs of metabolic decompensation."
        )


# ── Brand + presentation layer ─────────────────────────────────────────────
APP_CSS = """
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }

  /* Product masthead */
  .masthead {
      display: flex; align-items: center; justify-content: space-between;
      gap: 16px; flex-wrap: wrap; padding: 2px 0 15px;
      border-bottom: 1px solid #eaecf0;
  }
  .masthead .brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
  .masthead .logo { width: 46px; height: 46px; flex: 0 0 46px; display: block;
      filter: drop-shadow(0 2px 5px rgba(31,111,235,.18)); }
  .eyebrow {
      font-size: .65rem; font-weight: 700; letter-spacing: .14em;
      text-transform: uppercase; color: #667085; margin-bottom: 2px;
  }
  .masthead h1.wordmark {
      font-size: 1.58rem; line-height: 1.15; font-weight: 700;
      letter-spacing: -.025em; color: #101828; margin: 0; padding: 0;
  }
  .mastline { font-size: .77rem; color: #667085; margin-top: 3px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .chip {
      display: inline-flex; align-items: center; gap: 6px;
      font-size: .71rem; font-weight: 600; color: #344054;
      background: #f8fafc; border: 1px solid #e4e7ec;
      border-radius: 999px; padding: 4px 10px; white-space: nowrap;
  }
  .chip.accent { background: #eef4ff; border-color: #d6e4ff; color: #1d4ed8; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: #12b76a;
      box-shadow: 0 0 0 3px rgba(18,183,106,.12); }
  .lede { font-size: .86rem; color: #475467; margin: 13px 0 8px; max-width: 840px; }
  .bandline {
      display: flex; align-items: center; gap: 8px;
      font-size: 1.02rem; font-weight: 700; margin: -2px 0 3px;
  }
  .banddot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; }
  .bandcap { font-size: .78rem; color: #667085; margin-bottom: 10px; }

  /* Sign-in surface */
  .login-head { text-align: center; margin: 6vh 0 20px; }
  .login-logo { display: flex; justify-content: center; margin-bottom: 13px; }
  .login-head h1.login-title { font-size: 1.45rem; line-height: 1.25;
      font-weight: 700; letter-spacing: -.02em; color: #101828;
      margin: 0; padding: 0; }
  .login-sub { font-size: .85rem; color: #667085; margin-top: 6px; }
  .login-card-title { font-size: 1.02rem; font-weight: 700; color: #101828; }
  .login-foot { text-align: center; font-size: .72rem; color: #98a2b3;
      margin-top: 18px; }

  /* Existing component polish */
  div[data-testid="stMetric"] {
      background: #ffffff; border: 1px solid #e6e9ee;
      border-radius: 14px; padding: 14px 16px;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
  .pill {
      display: inline-block; margin: -8px 0 10px 0; padding: 2px 10px;
      border-radius: 999px; font-size: 0.70rem; font-weight: 600;
      letter-spacing: 0.02em; border: 1px solid transparent;
  }
  .pill span { font-weight: 400; opacity: 0.8; margin-left: 6px; }
  .tag {
      font-size: 0.72rem; font-weight: 600; padding: 2px 9px;
      border-radius: 999px; background: #eef1f4; color: #40474e;
      border: 1px solid #dfe3e8;
  }
  button[data-baseweb="tab"] {
      padding-left: 8px; padding-right: 8px; font-size: 0.85rem;
  }

  @media (max-width: 700px) {
      .masthead h1.wordmark { font-size: 1.28rem; }
      .masthead .logo { width: 40px; height: 40px; flex-basis: 40px; }
      .login-head { margin-top: 3vh; }
  }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


def brand_mark(size=46, uid="main"):
    """Inline SVG logo: a glucose droplet with an ECG trace on a blue tile."""
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 48 48" fill="none"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Diabetes Risk Predictor">
      <defs>
        <linearGradient id="logo-{uid}" x1="6" y1="4" x2="42" y2="46"
                        gradientUnits="userSpaceOnUse">
          <stop stop-color="#3B82F6"/>
          <stop offset="1" stop-color="#1D4ED8"/>
        </linearGradient>
      </defs>
      <rect width="48" height="48" rx="13" fill="url(#logo-{uid})"/>
      <rect x="0.6" y="0.6" width="46.8" height="46.8" rx="12.6"
            stroke="white" stroke-opacity="0.18"/>
      <path d="M24 10.5c4.3 5.4 8.2 9.9 8.2 14.9a8.2 8.2 0 1 1-16.4 0c0-5 3.9-9.5 8.2-14.9Z"
            fill="white" fill-opacity="0.95"/>
      <path d="M18.2 25.4h3.7l1.8-3.9 2.7 6.9 1.9-3.9h2.4"
            stroke="#1D4ED8" stroke-width="1.7" stroke-linecap="round"
            stroke-linejoin="round"/>
    </svg>
    """


# ── Auth ────────────────────────────────────────────────────────────────────
def login():
    if st.session_state.get("auth"):
        return True

    _, middle, _ = st.columns([1, 1.55, 1])
    with middle:
        st.markdown(
            f"""
            <div class="login-head">
              <div class="login-logo">{brand_mark(58, "login")}</div>
              <h1 class="login-title">Diabetes Risk Predictor</h1>
              <div class="login-sub">Clinical screening support for type 2 diabetes</div>
            </div>
            """,
            unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="login-card-title">Sign in</div>',
                        unsafe_allow_html=True)
            st.caption("Authorised clinical users only.")
            with st.form("login", border=False):
                u = st.text_input("Username", placeholder="e.g. clinician.name")
                p = st.text_input("Password", type="password")
                ok = st.form_submit_button("Sign in", type="primary",
                                           use_container_width=True)
        if ok:
            try:
                good_u = str(st.secrets["auth"]["username"])
                good_p = str(st.secrets["auth"]["password"])
            except Exception:
                good_u, good_p = "admin", "smit123"
            if hmac.compare_digest(u, good_u) and hmac.compare_digest(p, good_p):
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown(
            '<div class="login-foot">Educational screening aid &middot; '
            "not a diagnosis</div>",
            unsafe_allow_html=True)
    return False


if not login():
    st.stop()

# Stage 5 session state. `history` never leaves this browser session unless the
# user downloads it from the history tab.
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_label", "")


# --- Stage 3: shared presentation tokens -----------------------------------
# Status colours: (text, tint). Used by pills, the gauge and the summary table.
STATUS_STYLE = {
    "Normal":       ("#1a7f37", "#e6f4ea"),
    "Borderline":   ("#8a6100", "#fff4d6"),
    "Abnormal":     ("#b42318", "#fde8e6"),
    "Not measured": ("#57606a", "#eef1f4"),
}
STATUS_ICON = {"Normal": "●", "Borderline": "▲",
               "Abnormal": "■", "Not measured": "○"}
BAND_COLOUR = {"Low": "#2e9e4f", "Moderate": "#e0a223", "High": "#d64545"}

# Stage 4: risk factors a patient can realistically change, with the healthy
# target each scenario moves them to. Shared by Tab 1 and the recommendation
# engine so both always use the same targets.
MODIFIABLE = {
    "Glucose":       (100,  "mg/dL",  "fasting glucose around 100 mg/dL"),
    "BMI":           (23.0, "kg/m²",  "BMI inside the healthy 18.5–24.9 band"),
    "BloodPressure": (75,   "mm Hg",  "diastolic pressure around 75 mm Hg"),
    "Insulin":       (80,   "mu U/ml", "insulin in the middle of the normal range"),
    "SkinThickness": (20,   "mm",     "skin fold around 20 mm"),
}
EFFORT = {
    "Glucose":       "Diet + medication…",
    "BMI":           "Diet + exercise…",
    "BloodPressure": "Diet + medication…",
    "Insulin":       "Medical review…",
    "SkinThickness": "Nutrition review…",
}
COMPACT_LABELS = {
    "Glucose":       "Glucose",
    "BMI":           "BMI",
    "BloodPressure": "Blood pressure",
    "Insulin":       "Insulin",
    "SkinThickness": "Skin fold",
}



def field_status(feature, value):
    """Classify one input as Normal / Borderline / Abnormal / Not measured.

    Uses clinical bands rather than a single reference range, because e.g. a
    glucose of 150 is not "slightly above normal" for a screening tool. Values
    of 0 in the lab features mean "not measured", matching validate_inputs().
    """
    if feature in ZERO_AS_MISSING and (value is None or float(value) <= 0):
        return "Not measured"
    if value is None or pd.isna(value):
        return "Not measured"
    v = float(value)

    if feature == "Glucose":
        if v >= 200:
            return "Abnormal"          # diabetic range
        if v >= 140 or v < 70:
            return "Borderline"        # pre-diabetic / hypoglycaemic
        return "Normal"
    if feature == "BMI":
        if v >= 30:
            return "Abnormal"          # obese
        if v >= 25 or v < 18.5:
            return "Borderline"        # overweight / underweight
        return "Normal"
    if feature == "BloodPressure":
        if v >= 100 or v < 50:
            return "Abnormal"
        if v >= 90 or v < 60:
            return "Borderline"
        return "Normal"
    if feature == "Insulin":
        if v > 250:
            return "Abnormal"
        if v > 166 or v < 16:
            return "Borderline"
        return "Normal"
    if feature == "SkinThickness":
        if v > 45:
            return "Abnormal"
        if v > 30 or v < 10:
            return "Borderline"
        return "Normal"
    if feature == "Pregnancies":
        return "Borderline" if v > 5 else "Normal"
    if feature == "DiabetesPedigreeFunction":
        return "Borderline" if v >= 0.5 else "Normal"
    if feature == "Age":
        return "Borderline" if v >= 45 else "Normal"
    return "Normal"


def status_pill(feature, value):
    """Coloured HTML badge for one input (Streamlit cannot tint a slider itself)."""
    status = field_status(feature, value)
    text, tint = STATUS_STYLE[status]
    note = {
        "Normal": FEATURE_INFO[feature]["normal"],
        "Borderline": "outside the ideal band",
        "Abnormal": "clinically abnormal",
        "Not measured": "model will impute the median",
    }[status]
    return (f'<div class="pill" style="color:{text};background:{tint};'
            f'border-color:{text}33">{STATUS_ICON[status]} {status}'
            f'<span>{note}</span></div>')


def show_pill(feature, value):
    """Render the status badge directly under a sidebar control."""
    st.markdown(status_pill(feature, value), unsafe_allow_html=True)


def risk_gauge(risk, threshold, level):
    """Matplotlib speedometer: three risk bands, a needle and the threshold.

    Drawn with plain patches instead of Plotly so the gauge cannot fall back to
    a bare progress bar when an optional package is missing from a deployment.
    """
    fig = plt.figure(figsize=(4.6, 2.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-0.37, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

    # Wedge angles: 180° = 0%, 90° = 50%, 0° = 100%.
    for low_angle, high_angle, colour in ((126, 180, "#e6f4ea"),
                                          (72, 126, "#fff4d6"),
                                          (0, 72, "#fde8e6")):
        ax.add_patch(Wedge((0, 0), 1.0, low_angle, high_angle, width=0.22,
                           facecolor=colour, edgecolor="white", linewidth=1.5))

    needle = np.deg2rad(180 - 1.8 * risk * 100)
    ax.plot([0, 0.74 * np.cos(needle)], [0, 0.74 * np.sin(needle)],
            color=BAND_COLOUR[level], lw=4, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((0, 0), 0.055, color=BAND_COLOUR[level], zorder=5))

    tick = np.deg2rad(180 - 1.8 * threshold * 100)
    ax.plot([0.78 * np.cos(tick), 1.05 * np.cos(tick)],
            [0.78 * np.sin(tick), 1.05 * np.sin(tick)],
            color="#111827", lw=2.5, zorder=3)

    ax.text(0, -0.13, f"{risk:.1%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#1f2933")
    ax.text(0, -0.31, f"{level} risk", ha="center", va="center",
            fontsize=9, fontweight="600", color=BAND_COLOUR[level])

    for value in (0, 30, 60, 100):
        angle = np.deg2rad(180 - 1.8 * value)
        ax.text(1.13 * np.cos(angle), 1.13 * np.sin(angle), f"{value}%",
                ha="center", va="center", fontsize=8, color="#6b7280")

    fig.tight_layout(pad=0.1)
    return fig


# ── Stage 4: explanation + recommendation helpers ───────────────────────────
def risk_level(risk):
    """Map a probability to the three bands used across the app."""
    return "Low" if risk < .3 else "Moderate" if risk < .6 else "High"


def short_label(feature):
    """Feature name without its unit parenthesis, for compact tables/metrics."""
    return FEATURE_INFO[feature]["label"].split(" (")[0]


def tint_status(value):
    """Pandas Styler callback: colour a Status cell using the app's palette."""
    text, tint = STATUS_STYLE.get(value, ("#1f2933", "#ffffff"))
    return f"background-color: {tint}; color: {text}; font-weight: 600"


def build_explanation(x, cleaned, risk):
    """Plain-language explanation data using SHAP as a hidden engine.

    Returns a dict with headline, raises/lowers cards, range bars,
    try-change sentence, how-sure text, and a warning list. No raw SHAP
    values are exposed.
    """
    headline_fallback = False
    sv = None
    if explainer is not None:
        sv_result, err = safe(lambda: explainer(x), SHAP_ROW_TIMEOUT,
                              "SHAP explanation for this patient")
        if sv_result is not None:
            try:
                sv = np.asarray(sv_result.values)[0]
            except Exception:
                sv = None

    # Fixed importance order for fallback
    FIXED_ORDER = ["Glucose", "BMI", "Age", "Insulin", "BloodPressure",
                   "SkinThickness", "DiabetesPedigreeFunction", "Pregnancies"]

    if sv is not None:
        total_abs = float(np.sum(np.abs(sv))) + 1e-9
        shares = np.abs(sv) / total_abs
        features_in_order = list(np.argsort(-np.abs(sv)))
        raises = []
        lowers = []
        strength_map = {}
        for idx in features_in_order:
            feat = FEATURES[idx]
            share = float(shares[idx])
            if share >= 0.35:
                strength = "Strong"
            elif share >= 0.15:
                strength = "Moderate"
            else:
                strength = "Small"
            strength_map[feat] = strength
            status = field_status(feat, cleaned.get(feat))
            value = float(x.iloc[0][feat]) if feat in x.columns else None
            # Re-check missing from cleaned
            actual_value = cleaned.get(feat)
            is_missing = (feat in ZERO_AS_MISSING and (actual_value is None or (isinstance(actual_value, float) and np.isnan(actual_value)) or float(actual_value if actual_value is not None else -1) <= 0))
            direction = "raises" if float(sv[idx]) > 0 else "lowers"
            card = {
                "feature": feat,
                "label": FEATURE_INFO[feat]["label"],
                "value": fmt_value(feat, actual_value) if not is_missing else "Not measured",
                "status": status,
                "strength": f"{strength} ({share * 100:.0f}% of decision)",
                "direction": direction,
                "sentence": (explain_missing(feat, MEDIANS.get(feat, 100))
                              if is_missing else
                              (explain_feature_up(feat, float(actual_value)) if direction == "raises" else explain_feature_down(feat, float(actual_value)))),
                "imputed_note": ("Estimated, not measured — model filled in the median." if is_missing else ""),
            }
            if direction == "raises":
                raises.append(card)
            else:
                lowers.append(card)
        # Headline: top two drivers by |SHAP|
        sorted_by_abs = sorted(features_in_order, key=lambda i: -abs(float(sv[i])))
        top_two = [FEATURES[i] for i in sorted_by_abs[:2]]
        headline = (f"Risk is {risk:.0%} ({risk_level(risk)}). "
                    f"{FEATURE_INFO[top_two[0]]['label'].split(' (')[0]} and "
                    f"{FEATURE_INFO[top_two[1]]['label'].split(' (')[0]} matter most for this result.")
    else:
        headline_fallback = True
        # Fallback: fixed order + status-based direction
        raises = []
        lowers = []
        strength_map = {}
        for feat in FIXED_ORDER:
            status = field_status(feat, cleaned.get(feat))
            is_missing = feat in ZERO_AS_MISSING and (cleaned.get(feat) is None or (isinstance(cleaned.get(feat), float) and np.isnan(cleaned.get(feat))) or float(cleaned.get(feat) if cleaned.get(feat) is not None else -1) <= 0)
            # Direction heuristic: abnormal / borderline / high value -> raises; else lowers
            value = cleaned.get(feat)
            try:
                val_f = float(value) if value is not None and not np.isnan(value) else None
            except Exception:
                val_f = None
            if is_missing:
                direction = "raises"  # missing is treated cautiously as risk contributor
            elif status in ("Abnormal", "Borderline"):
                direction = "raises"
            else:
                direction = "lowers"
            card = {
                "feature": feat,
                "label": FEATURE_INFO[feat]["label"],
                "value": fmt_value(feat, value) if not is_missing else "Not measured",
                "status": status,
                "strength": "Moderate",
                "direction": direction,
                "sentence": (explain_missing(feat, MEDIANS.get(feat, 100))
                              if is_missing else
                              (explain_feature_up(feat, val_f) if direction == "raises" else explain_feature_down(feat, val_f))),
                "imputed_note": ("Estimated, not measured — model filled in the median." if is_missing else ""),
            }
            strength_map[feat] = "Moderate"
            if direction == "raises":
                raises.append(card)
            else:
                lowers.append(card)
        # Pick headline from first abnormal/borderline feature, else first in order
        headline_features = [feat for feat in FIXED_ORDER if field_status(feat, cleaned.get(feat)) in ("Abnormal", "Borderline")]
        if not headline_features:
            headline_features = FIXED_ORDER[:2]
        headline = (f"Risk is {risk:.0%} ({risk_level(risk)}). "
                    f"{FEATURE_INFO[headline_features[0]]['label'].split(' (')[0]} and "
                    f"{FEATURE_INFO[headline_features[1]]['label'].split(' (')[0]} are the key factors.")

    # Range bars for measurements
    range_bars = {}
    for feat in FEATURES:
        value = cleaned.get(feat)
        try:
            val_f = float(value) if value is not None and not np.isnan(value) else None
        except Exception:
            val_f = None
        if feat == "Glucose":
            normal_low, normal_high = 70, 140
        elif feat == "BMI":
            normal_low, normal_high = 18.5, 24.9
        elif feat == "BloodPressure":
            normal_low, normal_high = 60, 80
        elif feat == "SkinThickness":
            normal_low, normal_high = 10, 30
        elif feat == "Insulin":
            normal_low, normal_high = 16, 166
        elif feat == "Pregnancies":
            normal_low, normal_high = 0, 5
        elif feat == "DiabetesPedigreeFunction":
            normal_low, normal_high = 0, 0.5
        elif feat == "Age":
            normal_low, normal_high = 21, 45
        else:
            normal_low, normal_high = 0, 1
        range_bars[feat] = {
            "label": FEATURE_INFO[feat]["label"],
            "value": fmt_value(feat, value) if val_f is not None else "Not measured",
            "normal_low": normal_low,
            "normal_high": normal_high,
            "current": val_f,
        }

    # How sure
    interval_text = ""
    # We'll return a simple text that the tab will format using prediction_interval
    how_sure_text = "The model's estimate comes with a range of possible outcomes for similar patients."

    # Try-change sentence (placeholder, the tab handles interaction)
    try_change_sentence = "Changing one modifiable value can shift the estimated risk."

    # Warning list
    missing_features_local = []
    for feat in ZERO_AS_MISSING:
        v = cleaned.get(feat)
        if v is None or (isinstance(v, float) and np.isnan(v)) or float(v if v is not None else -1) <= 0:
            missing_features_local.append(feat)

    warning_text = (f"{len(missing_features_local)} measurement(s) were not recorded and were estimated from the dataset median: "
                    + ", ".join(FEATURE_INFO[f]["label"] for f in missing_features_local) + "."
                    if missing_features_local else "All measurements were provided.")

    return {
        "headline": headline,
        "headline_fallback": headline_fallback,
        "raises": raises[:3],
        "lowers": lowers[:3],
        "range_bars": range_bars,
        "how_sure_text": how_sure_text,
        "warning_text": warning_text,
        "missing_features": missing_features_local,
        "strength_map": strength_map,
        "try_change_sentence": try_change_sentence,
    }


def improvement_scenarios(cleaned_row, model_obj):
    """Rank modifiable values by the risk left after moving each to its target.

    Returns (rows, combined_risk). Each row carries the current value, target,
    resulting risk and reduction. The combined scenario moves every eligible
    value at once and is written with a caveat in the UI, because real risk
    factors are correlated and the model treats them independently.
    """
    rows = []
    for feature, (target, unit, note) in MODIFIABLE.items():
        current = cleaned_row.get(feature)
        if current is None or pd.isna(current):
            continue
        try:
            current = float(current)
        except (TypeError, ValueError):
            continue
        if current <= float(target):
            continue
        try:
            changed = cleaned_row.copy()
            changed[feature] = float(target)
            new_risk = float(model_obj.predict_proba(
                pd.DataFrame([changed])[FEATURES])[0, 1])
        except Exception:  # noqa: BLE001
            continue
        rows.append({
            "feature": feature,
            "current": current,
            "target": float(target),
            "unit": unit,
            "note": note,
            "new_risk": min(max(new_risk, 0.0), 1.0),
        })

    rows.sort(key=lambda row: row["new_risk"])
    for position, row in enumerate(rows, start=1):
        row["rank"] = position

    combined_risk = None
    if rows:
        combo = cleaned_row.copy()
        for row in rows:
            combo[row["feature"]] = row["target"]
        try:
            combined_risk = float(model_obj.predict_proba(
                pd.DataFrame([combo])[FEATURES])[0, 1])
            combined_risk = min(max(combined_risk, 0.0), 1.0)
        except Exception:  # noqa: BLE001
            combined_risk = None
    return rows, combined_risk


# ── Stage 5: saved history, batch scoring, reports, view modes ─────────────
BATCH_LIMIT = 2000


def make_history_record(risk, cleaned, missing_features, label, note=""):
    """One saved prediction. Values are stored as plain floats (NaN → None)."""
    return {
        "id": uuid.uuid4().hex[:8],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "label": (label or "").strip() or "Unnamed patient",
        "risk": round(float(risk), 4),
        "level": risk_level(risk),
        "decision": "Positive" if risk >= THRESHOLD else "Negative",
        "values": {
            feature: (None if pd.isna(cleaned[feature]) else float(cleaned[feature]))
            for feature in FEATURES
        },
        "missing": list(missing_features),
        "note": (note or "").strip(),
    }


def history_frame(history):
    """Display frame for saved records, newest first."""
    rows = []
    for record in reversed(history):
        try:
            when = datetime.fromisoformat(str(record["saved_at"]))
            stamp = when.strftime("%d %b %Y, %H:%M")
        except (TypeError, ValueError):
            stamp = str(record.get("saved_at", ""))
        rows.append({
            "Saved": stamp,
            "Patient": record.get("label", "Unnamed patient"),
            "Risk": f"{float(record.get('risk', 0)):.1%}",
            "Band": record.get("level", ""),
            "Screening call": record.get("decision", ""),
            "Note": record.get("note", ""),
        })
    return pd.DataFrame(rows)


def shap_drivers(row_frame, k=3):
    """Top-k SHAP drivers as [(feature, value)], or None when unavailable."""
    if explainer is None:
        return None
    values, _ = safe(lambda: explainer(row_frame), SHAP_ROW_TIMEOUT,
                     "SHAP explanation for this patient")
    if values is None:
        return None
    try:
        vector = np.asarray(values.values)[0]
        order = np.argsort(-np.abs(vector))[:k]
        return [(FEATURES[i], float(vector[i])) for i in order]
    except Exception:  # noqa: BLE001
        return None


def save_prediction_panel(risk, cleaned, missing_features, prefix):
    """Save-to-history card used by the clinician view."""
    with st.container(border=True):
        st.markdown("#### Save this prediction")
        st.caption("Kept in this app session's memory. Download the history from "
                   "the History tab to keep a copy.")
        label_col, note_col = st.columns([1, 2], gap="large")
        label = label_col.text_input(
            "Patient name or ID", value=st.session_state.get("last_label", ""),
            key=f"{prefix}_save_label")
        note = note_col.text_input(
            "Note (optional)", key=f"{prefix}_save_note",
            placeholder="e.g. repeat HbA1c in 3 months")
        if st.button("Save to history", key=f"{prefix}_save_btn"):
            final_label = (label.strip() or st.session_state.get("last_label")
                           or "Unnamed patient")
            st.session_state["last_label"] = final_label
            st.session_state["history"].append(
                make_history_record(risk, cleaned, missing_features,
                                    final_label, note))
            st.success(f"Saved **{final_label}** at **{risk:.1%}**. "
                       "Open the History tab to see the trend.")


def render_history(prefix, compact=False, only_label=None):
    """History table, trend chart and export/delete controls."""
    history = st.session_state.get("history", [])
    if only_label:
        history = [row for row in history if row.get("label") == only_label]
    if not history:
        st.info("No saved predictions yet. Save one and it will appear here "
                "with a trend chart."
                if not only_label else
                f"No saved predictions for **{only_label}** yet.")
        return

    frame = history_frame(history)
    if not compact:
        filter_col, _ = st.columns([1, 2])
        labels = sorted({row.get("label", "") for row in history})
        chosen = filter_col.selectbox(
            "Filter by patient", ["All patients"] + labels, key=f"{prefix}_filter")
        if chosen != "All patients":
            frame = frame[frame["Patient"] == chosen]
            history = [row for row in history if row.get("label") == chosen]

    patients = frame["Patient"].nunique()
    if patients == 1 and len(frame) >= 2:
        risks = [float(row["risk"]) for row in history]
        latest, first = risks[-1], risks[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Saved predictions", len(frame))
        m2.metric("Latest risk", f"{latest:.1%}")
        m3.metric("Change since first", f"{(latest - first) * 100:+.1f} pts")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Saved predictions", len(frame))
        m2.metric("Patients tracked", patients)

    st.markdown("#### Risk over time")
    fig = history_trend_figure(
        [{"saved_at": row.get("saved_at"), "risk": row.get("risk"),
          "label": row.get("label")} for row in history],
        threshold=THRESHOLD)
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("The saved timestamps could not be plotted.")

    st.dataframe(frame, hide_index=True, use_container_width=True)

    export_col, json_col = st.columns(2)
    export_col.download_button(
        "⬇️ History CSV", frame.to_csv(index=False).encode("utf-8"),
        "patient_history.csv", "text/csv", key=f"{prefix}_csv")
    json_col.download_button(
        "⬇️ History JSON",
        json.dumps(history, indent=2, ensure_ascii=False).encode("utf-8"),
        "patient_history.json", "application/json", key=f"{prefix}_json")

    if compact:
        return

    with st.expander("🗑️ Manage saved records"):
        st.caption("Deleting a record cannot be undone. Downloads above are "
                   "your permanent copy.")
        entries = []
        for record in reversed(history):
            try:
                stamp = datetime.fromisoformat(
                    str(record["saved_at"])).strftime("%d %b %Y, %H:%M")
            except (TypeError, ValueError):
                stamp = str(record.get("saved_at", ""))
            entries.append((f"{stamp} · {record.get('label', '')} · "
                            f"{float(record.get('risk', 0)):.1%} · "
                            f"#{record.get('id', '')}",
                            record.get("id")))
        mapping = dict(entries)
        picked = st.multiselect("Records to delete", list(mapping),
                                key=f"{prefix}_delete_pick")
        confirm = st.checkbox("I understand these records will be removed",
                              key=f"{prefix}_delete_confirm")
        if st.button("Delete selected", key=f"{prefix}_delete_btn",
                     disabled=not (picked and confirm)):
            chosen_ids = {mapping[item] for item in picked}
            st.session_state["history"] = [
                row for row in st.session_state["history"]
                if row.get("id") not in chosen_ids]
            st.rerun()
        if st.button("Clear all history", key=f"{prefix}_clear_btn",
                     disabled=not confirm):
            st.session_state["history"] = []
            st.rerun()


def normalise_batch(frame):
    """Match CSV columns to FEATURES, coerce numbers and apply 0 → missing."""
    data, missing_columns = {}, []
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for feature in FEATURES:
        source = lookup.get(feature.lower())
        if source is None:
            missing_columns.append(feature)
            data[feature] = pd.Series(np.nan, index=frame.index, dtype=float)
        else:
            data[feature] = pd.to_numeric(frame[source], errors="coerce")
    normalised = pd.DataFrame(data)
    for feature in ZERO_AS_MISSING:
        normalised.loc[normalised[feature] == 0, feature] = np.nan
    return normalised, missing_columns


def batch_predict(frame, model_obj, limit=BATCH_LIMIT):
    """Score a batch frame. Returns (results, probabilities, notes)."""
    if frame is None or len(frame) == 0:
        raise ValueError("The file has no rows to score.")
    notes = []
    if len(frame) > limit:
        frame = frame.head(limit).copy()
        notes.append(f"Only the first **{limit}** rows were scored; split larger "
                     "files and run them in separate batches.")

    normalised, missing_columns = normalise_batch(frame)
    if missing_columns:
        notes.append("These columns were absent and filled with the dataset "
                     "median: **" + ", ".join(missing_columns) + "**.")
    for feature in FEATURES:
        if normalised[feature].isna().all():
            normalised[feature] = ALL_MEDIANS[feature]

    probabilities = np.asarray(
        model_obj.predict_proba(normalised[FEATURES])[:, 1], dtype=float)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    frame = frame.reset_index(drop=True)
    results = pd.concat([
        pd.DataFrame({
            "Risk %": np.round(probabilities * 100, 1),
            "Band": [risk_level(value) for value in probabilities],
            "Screening call": [
                "Positive" if value >= THRESHOLD else "Negative"
                for value in probabilities],
            "Missing labs": normalised[ZERO_AS_MISSING].isna().sum(
                axis=1).astype(int).to_numpy(),
        }, index=frame.index),
        frame,
    ], axis=1)
    return results, probabilities, notes


def render_report_section(prefix, *, risk, cleaned, interval, flags, drivers,
                          default_label="", label_override=None):
    """PDF preview + download. Shared by the clinician tab and patient view."""
    st.markdown(
        "The report packages the risk, the measurements, the main drivers, the "
        "recommended next steps, clinical flags and any saved trend. It is a "
        "screening summary to discuss with a clinician, not a diagnosis.")

    if label_override is None:
        label_col, note_col = st.columns([1, 2], gap="large")
        label = label_col.text_input(
            "Patient name or ID", value=default_label or st.session_state.get("last_label", ""),
            key=f"{prefix}_label")
        note = note_col.text_area(
            "Clinician note for the report (optional)", key=f"{prefix}_note",
            height=80, placeholder="Assessment, plan, follow-up…")
        include_history = st.checkbox(
            "Include saved progress chart", value=True,
            key=f"{prefix}_include_history")
    else:
        label = label_override
        note = st.text_area("Note for the report (optional)",
                            key=f"{prefix}_note", height=80)
        include_history = True

    if not PDF_OK:
        st.warning("PDF export needs the `reportlab` package. Add "
                   "`reportlab>=4.0.0` to `requirements.txt` and reboot the app.")
        return

    records = list(st.session_state.get("history", []))
    wanted_label = label.strip() or "Unnamed patient"
    records = [row for row in records if row.get("label") == wanted_label]
    if not include_history:
        records = []

    rec_rows, combined_risk = improvement_scenarios(cleaned, model)
    value_rows = [
        (FEATURE_INFO[feature]["label"], fmt_value(feature, cleaned[feature]),
         FEATURE_INFO[feature]["normal"], field_status(feature, cleaned[feature]))
        for feature in FEATURES
    ]
    driver_rows = ([(FEATURE_INFO[feature]["label"], value)
                    for feature, value in drivers]
                   if drivers else None)
    pdf_recs = [{
        "feature": FEATURE_INFO[row["feature"]]["label"],
        "current": row["current"],
        "target": row["target"],
        "unit": row["unit"],
        "new_risk": row["new_risk"],
    } for row in rec_rows]

    generated = datetime.now()
    try:
        pdf_bytes = build_report_pdf(
            label=label.strip() or "Unnamed patient",
            risk=risk, level=risk_level(risk), threshold=THRESHOLD,
            decision="Positive" if risk >= THRESHOLD else "Negative",
            interval=interval, values=value_rows, rec_rows=pdf_recs,
            combined_risk=combined_risk, flags=flags, drivers=driver_rows,
            history_rows=records, note=note, model_name=meta.get("best_model", ""),
            generated_at=generated)
    except Exception:  # noqa: BLE001
        st.error("❌ The PDF could not be generated. The risk score is unaffected.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        return

    if not pdf_bytes:
        st.warning("PDF export is unavailable in this environment.")
        return

    summary = st.columns(3)
    summary[0].metric("Report risk", f"{risk:.1%}")
    summary[1].metric("Band", risk_level(risk))
    summary[2].metric("Saved visits in report", len(records))

    slug = "".join(ch if ch.isalnum() else "_" for ch in (label or "patient"))
    slug = slug.strip("_")[:40] or "patient"
    st.download_button(
        "⬇️ Download PDF report", pdf_bytes,
        file_name=f"diabetes_report_{slug}_{generated:%Y%m%d_%H%M}.pdf",
        mime="application/pdf", key=f"{prefix}_pdf")
    st.caption(f"{len(pdf_bytes) / 1024:.0f} KB · generated "
               f"{generated:%d %b %Y, %H:%M} · built in this session, "
               "not stored on the server.")


def render_patient_view(*, risk, cleaned, missing_features, interval, flags,
                        rec_rows, combined_risk, drivers):
    """Plain-language view: no model internals, no raw SHAP values."""
    level = risk_level(risk)
    name = st.text_input(
        "Your name or ID (optional)", value=st.session_state.get("last_label", ""),
        key="patient_name",
        help="Used only to label your saved results and report. It is not sent anywhere.")
    if name.strip():
        st.session_state["last_label"] = name.strip()

    st.markdown("### Your result")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        with st.container(border=True):
            fig = risk_gauge(risk, THRESHOLD, level)
            st.pyplot(fig)
            plt.close(fig)
            if interval:
                st.caption(
                    f"People with very similar measurements scored between "
                    f"**{interval[0]:.0%}** and **{interval[1]:.0%}**. That range "
                    "is the honest uncertainty around this single number.")
    with right:
        with st.container(border=True):
            st.markdown(
                f'<div class="bandline" style="font-size:1.45rem;'
                f'margin:0 0 6px;color:{BAND_COLOUR[level]}">'
                f'<span class="banddot" style="background:{BAND_COLOUR[level]}"></span>'
                f'{level} risk</div>', unsafe_allow_html=True)
            st.markdown(
                f"About **{risk * 100:.0f} in 100** people with measurements like "
                "yours would be expected to have diabetes.")
            if risk >= THRESHOLD:
                st.warning(
                    "Your result is at or above the screening cut-off. That is a "
                    "reason to see a doctor for a confirmatory test — it is **not** "
                    "a diagnosis.")
            else:
                st.success(
                    "Your result is below the screening cut-off. Keep up your "
                    "current habits, and re-screen if your measurements change.")
            if missing_features:
                st.caption(
                    f"{len(missing_features)} of the 8 measurements were not "
                    "available, so this estimate is rougher than usual.")

    # Patient-view explanation section (reuses build_explanation, shorter wording)
    try:
        expl_pat = build_explanation(x, cleaned, risk)
        st.markdown("---")
        st.markdown("### Why this result")
        with st.container(border=True):
            st.markdown(synthesize_clinical_narrative(cleaned, risk))
            cohort_line = similar_patients_summary(x)
            if cohort_line:
                st.caption(cohort_line)
        for card in expl_pat["raises"][:2]:
            with st.container(border=True):
                st.markdown(f"**{card['label']}** · {card['value']} · {card['status']} · raises risk")
                st.markdown(card["sentence"])
        for card in expl_pat["lowers"][:2]:
            with st.container(border=True):
                st.markdown(f"**{card['label']}** · {card['value']} · {card['status']} · lowers risk")
                st.markdown(card["sentence"])
        st.caption("These are the main factors behind your screening estimate. Talk through any that concern you with your doctor.")
    except Exception:
        st.caption("A brief explanation of the main drivers is not available right now, but the risk score above is still valid.")

    st.markdown("#### What this number means")
    st.markdown(
        "- It is a **screening estimate** from eight routine measurements, not a "
        "diagnosis.\n"
        "- A lower score is reassuring but does **not** rule out diabetes.\n"
        "- A higher score does **not** mean you have diabetes; it means a "
        "confirmatory blood test is sensible.\n"
        "- Risk changes as measurements change — that is why repeat screening "
        "matters.")

    st.markdown("#### What you can do next")
    actionable = [row for row in (rec_rows or [])
                  if (risk - row["new_risk"]) * 100 > 0.05]
    if actionable:
        for row in actionable[:3]:
            gain = (risk - row["new_risk"]) * 100
            point_word = "point" if round(gain) == 1 else "points"
            st.markdown(
                f"- **{COMPACT_LABELS[row['feature']]}**: from "
                f"{fmt_value(row['feature'], row['current'])} {row['unit']} to "
                f"**{row['target']:g} {row['unit']}** — about "
                f"**{gain:.0f} {point_word} lower** in this model.")
        if combined_risk is not None:
            st.caption(
                f"If every target above is reached together, the estimate falls "
                f"to about **{combined_risk:.0%}**. Treat that as an optimistic "
                "ceiling, not a promise.")
        st.caption("These are model estimates from changing one measurement at a "
                   "time. Your clinician can turn them into a real care plan.")
    else:
        st.success("Every modifiable measurement is already in the healthy range. "
                   "Maintaining these values is the best next step.")

    with st.container(border=True):
        st.markdown("#### Worth mentioning to your doctor")
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.caption("No clinical thresholds were exceeded in the values entered.")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Save this result")
        note = st.text_input("Add a note (optional)", key="patient_note",
                             placeholder="e.g. after starting a new diet")
        if st.button("Save to my progress", key="patient_save"):
            final_label = (name.strip() or st.session_state.get("last_label")
                           or "Unnamed patient")
            st.session_state["last_label"] = final_label
            st.session_state["history"].append(
                make_history_record(risk, cleaned, missing_features,
                                    final_label, note))
            st.success(f"Saved at **{risk:.1%}**. Your progress chart now includes "
                       "this visit.")

    st.markdown("### Your progress")
    render_history("patient_hist", compact=True,
                   only_label=name.strip() if name.strip() else None)

    st.markdown("### Your report")
    render_report_section(
        "patient_report", risk=risk, cleaned=cleaned, interval=interval,
        flags=flags, drivers=drivers,
        label_override=name.strip() or "Unnamed patient")

    st.caption("This tool is for education and screening support only. It does "
               "not replace a clinician's assessment or a laboratory diagnosis.")


# ── Cached resources ─────────────────────────────────────────────────────────
def needs_retrain():
    for path in (MODEL_PATH, METRICS_PATH, BG_PATH):
        if not os.path.exists(path):
            return True
    try:
        with open(METRICS_PATH, encoding="utf-8") as handle:
            saved_meta = json.load(handle)
        # A model pickled by a different scikit-learn build can silently return
        # wrong numbers, so treat a version mismatch as "needs retraining".
        if saved_meta.get("sklearn_version") != sklearn.__version__:
            return True
        if saved_meta.get("threshold") is None:
            return True   # artefacts from before threshold tuning existed
        saved_xgb = saved_meta.get("xgboost_version")
        if saved_xgb and saved_xgb != XGBOOST_VERSION:
            return True
        model = joblib.load(MODEL_PATH)
        bg = pd.read_csv(BG_PATH)
        if set(FEATURES) - set(bg.columns):
            return True
        dummy = pd.DataFrame(
            [[1, 120, 70, 25, 80, 28.0, 0.4, 35]], columns=FEATURES)
        model.predict_proba(dummy)
        return False
    except Exception:
        return True


@st.cache_resource(show_spinner=False)
def load():
    """Load (or train) the model, metrics, background sample and SHAP explainer."""
    if needs_retrain():
        st.info("🔄 No usable saved model was found, so one is being trained now.")
        with st.spinner("Training model — comparing four algorithms. This takes about 60 seconds…"):
            try:
                run_training()
            except TrainingError as exc:
                st.error(f"❌ Model training failed.\n\n{exc}")
                st.stop()
        st.success("Model trained and saved.")

    try:
        model = joblib.load(MODEL_PATH)
        with open(METRICS_PATH, encoding="utf-8") as handle:
            meta = json.load(handle)
        bg = pd.read_csv(BG_PATH)
    except Exception:
        st.error("❌ The saved model files could not be loaded.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()

    if set(FEATURES) - set(bg.columns):
        st.error(f"❌ `background.csv` is missing columns: {sorted(set(FEATURES) - set(bg.columns))}")
        st.stop()

    def predict_proba_fn(X):
        return model.predict_proba(pd.DataFrame(X, columns=FEATURES))[:, 1]

    # SHAP is optional: if it fails or times out the rest of the app keeps working.
    explainer, explainer_error = None, None
    with st.spinner("Preparing explainability engine…"):
        explainer, explainer_error = safe(
            lambda: shap.Explainer(predict_proba_fn, bg, feature_names=FEATURES),
            SHAP_INIT_TIMEOUT,
            "SHAP explainer setup",
        )

    return model, meta, bg, explainer, explainer_error


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


model, meta, bg, explainer, explainer_error = load()

# Medians used to fill "not measured" values, shown to the user for transparency.
MEDIANS = {
    feature: float(bg[feature].replace(0, np.nan).median())
    for feature in ZERO_AS_MISSING
}

# Every feature median, used only to fill a batch column that is entirely blank
# so the model pipeline always receives a usable frame.
ALL_MEDIANS = {
    feature: float(
        bg[feature].replace(0, np.nan).median()
        if feature in ZERO_AS_MISSING else bg[feature].median())
    for feature in FEATURES
}

# ── Decision threshold (tuned for recall in train_model.py) ─────────────────
THRESHOLD = float(meta["threshold"]) if meta.get("threshold") is not None else 0.5
THRESHOLD_RULE = meta.get("threshold_rule", "default 0.50")
AT_THRESHOLD = meta.get("at_threshold") or {}
AT_DEFAULT = meta.get("at_default") or {}

# Predicted risk for every background row — computed once, used by the
# uncertainty interval below so each slider move stays instant.
try:
    BG_RISKS = model.predict_proba(bg[FEATURES])[:, 1]
except Exception:  # noqa: BLE001
    BG_RISKS = None


def _standardise_background(bg_frame: pd.DataFrame):
    """Column-wise standardisation of the 100-row background sample."""
    filled = bg_frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    medians = filled.median()
    filled = filled.fillna(medians)          # DataFrame.fillna(Series) aligns on columns
    centres = filled.mean()
    spread = filled.std(ddof=0).replace(0, 1.0)
    return ((filled - centres) / spread).to_numpy(dtype=float), medians, spread


try:
    NEIGHBOUR_MATRIX, NEIGHBOUR_MEDIANS, NEIGHBOUR_SPREAD = _standardise_background(bg)
except Exception:  # noqa: BLE001
    NEIGHBOUR_MATRIX = None


def prediction_interval(x_row, risk, k=15):
    """Approximate 90% interval around `risk`, scaled by local heterogeneity.

    Uses the spread of predicted risk among the k training patients closest to
    this profile: a profile surrounded by very different patients gets a wide
    interval, one sitting in a tight cluster gets a narrow one. It is a
    prediction interval for *comparable patients*, not a sampling-error interval
    for the model itself.

    Returns (low, high, n_neighbours) or None if it cannot be computed.
    """
    if BG_RISKS is None or NEIGHBOUR_MATRIX is None or len(NEIGHBOUR_MATRIX) < 5:
        return None
    try:
        # A single row: fill NaN column-wise, then standardise with the same
        # statistics used for the background sample.
        values = x_row[FEATURES].astype(float).iloc[0].fillna(NEIGHBOUR_MEDIANS)
        row = ((values - NEIGHBOUR_MEDIANS) / NEIGHBOUR_SPREAD).to_numpy(dtype=float)
        k = max(5, min(k, len(NEIGHBOUR_MATRIX)))
        distances = np.sqrt(((NEIGHBOUR_MATRIX - row) ** 2).sum(axis=1))
        nearest = np.argsort(distances)[:k]
        probs = BG_RISKS[nearest]

        spread = float(np.std(probs, ddof=1)) if len(probs) > 1 else 0.0
        half_width = 1.645 * spread              # ~90% for a normal approximation
        low = max(0.0, risk - half_width)
        high = min(1.0, risk + half_width)
        return float(low), float(high), int(k)
    except Exception:  # noqa: BLE001
        return None


def similar_patients_summary(x_row, k=15):
    """Return a plain-English sentence about the k closest reference profiles."""
    if BG_RISKS is None or NEIGHBOUR_MATRIX is None or len(NEIGHBOUR_MATRIX) < 5:
        return None
    try:
        values = x_row[FEATURES].astype(float).iloc[0].fillna(NEIGHBOUR_MEDIANS)
        row = ((values - NEIGHBOUR_MEDIANS) / NEIGHBOUR_SPREAD).to_numpy(dtype=float)
        k = max(5, min(k, len(NEIGHBOUR_MATRIX)))
        distances = np.sqrt(((NEIGHBOUR_MATRIX - row) ** 2).sum(axis=1))
        nearest = np.argsort(distances)[:k]
        probs = BG_RISKS[nearest]
        flagged = int((probs >= THRESHOLD).sum())
        avg_prob = float(np.mean(probs))
        return (
            f"Among the **{k} most similar patients** in our reference cohort, "
            f"**{flagged} of {k}** scored above the screening threshold "
            f"(average risk **{avg_prob:.0%}**, range **{probs.min():.0%} – {probs.max():.0%}**)."
        )
    except Exception:  # noqa: BLE001
        return None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    mode = st.radio(
        "View mode", ["Clinician", "Patient"], index=0, horizontal=True,
        format_func=lambda value: value,
        help="Patient view hides the model internals and technical charts. "
             "Clinician view adds explainability charts, model performance, the data "
             "explorer, batch scoring and side-by-side comparison.")
    st.caption("Patient view: plain-language result and next steps. "
               "Clinician view: the full technical workspace.")
    st.markdown("---")

    st.header("Patient information")
    st.caption("Match the controls to the patient's measurements. Every input is "
               "colour-coded: **green** normal, **amber** borderline, **red** abnormal.")
    st.markdown("---")

    vals = {}
    vals["Pregnancies"] = st.number_input(
        FEATURE_INFO["Pregnancies"]["label"], min_value=0, max_value=20, value=1, step=1,
        help=FEATURE_INFO["Pregnancies"]["help"])
    show_pill("Pregnancies", vals["Pregnancies"])

    vals["Glucose"] = st.slider(
        FEATURE_INFO["Glucose"]["label"], 0, 250, 120,
        help=FEATURE_INFO["Glucose"]["help"] + SENTINEL_HELP)
    show_pill("Glucose", vals["Glucose"])

    vals["BloodPressure"] = st.slider(
        FEATURE_INFO["BloodPressure"]["label"], 0, 130, 70,
        help=FEATURE_INFO["BloodPressure"]["help"] + SENTINEL_HELP)
    show_pill("BloodPressure", vals["BloodPressure"])

    vals["SkinThickness"] = st.slider(
        FEATURE_INFO["SkinThickness"]["label"], 0, 100, 25,
        help=FEATURE_INFO["SkinThickness"]["help"] + SENTINEL_HELP)
    show_pill("SkinThickness", vals["SkinThickness"])

    vals["Insulin"] = st.slider(
        FEATURE_INFO["Insulin"]["label"], 0, 850, 80,
        help=FEATURE_INFO["Insulin"]["help"] + SENTINEL_HELP)
    show_pill("Insulin", vals["Insulin"])

    vals["BMI"] = st.slider(
        FEATURE_INFO["BMI"]["label"], 0.0, 70.0, 28.0, 0.1,
        help=FEATURE_INFO["BMI"]["help"] + SENTINEL_HELP)
    show_pill("BMI", vals["BMI"])

    vals["DiabetesPedigreeFunction"] = st.slider(
        FEATURE_INFO["DiabetesPedigreeFunction"]["label"], 0.05, 2.5, 0.4, 0.01,
        help=FEATURE_INFO["DiabetesPedigreeFunction"]["help"])
    show_pill("DiabetesPedigreeFunction", vals["DiabetesPedigreeFunction"])

    vals["Age"] = st.slider(
        FEATURE_INFO["Age"]["label"], 18, 90, 35,
        help=FEATURE_INFO["Age"]["help"])
    show_pill("Age", vals["Age"])

    cleaned, issues, missing_features = validate_inputs(vals, MEDIANS)

    # Compact tally so the user knows how many values need attention.
    tally = {}
    for feature in FEATURES:
        status = field_status(feature, vals[feature])
        tally[status] = tally.get(status, 0) + 1
    if tally.get("Abnormal"):
        st.error(f"■ {tally['Abnormal']} abnormal value(s) entered.")
    elif tally.get("Borderline"):
        st.warning(f"▲ {tally['Borderline']} borderline value(s) entered.")
    else:
        st.success("All entered values are in the normal band.")

    st.markdown("---")
    st.markdown("**Input checks**")
    if issues:
        for level, message in issues:
            {"warning": st.warning, "error": st.error, "info": st.info}[level](message)
    else:
        st.caption("No implausible or missing measurements detected.")

    with st.expander("Reference ranges"):
        for feature in FEATURES:
            st.caption(f"**{feature}**: {FEATURE_INFO[feature]['normal']}")

    st.markdown("---")
    st.caption(f"Model in use: **{meta.get('best_model', 'unknown')}** · "
               f"decision threshold **{THRESHOLD:.2f}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
if mode == "Patient":
    eyebrow = "Patient summary"
    heading = "Your diabetes risk results"
    mastline = ("Eight routine measurements &middot; plain-language summary")
    chips = ('<span class="chip accent">Screening estimate</span>'
             '<span class="chip">Not a diagnosis</span>')
    lede = ("This page turns your eight routine measurements into a plain-language "
            "estimate of diabetes risk. It supports a conversation with your "
            "clinician.")
else:
    eyebrow = "Clinical screening support"
    heading = "Diabetes Risk Predictor"
    mastline = ("Eight routine measurements &middot; recall-tuned screening")
    chips = (f'<span class="chip accent"><span class="dot"></span>Clinician workspace</span>'
             f'<span class="chip">{meta.get("best_model", "unknown")}</span>'
             f'<span class="chip">Threshold {THRESHOLD:.2f}</span>')
    lede = ("Estimates diabetes risk from eight routine measurements and shows "
            "why. Educational screening support — not a diagnosis.")

st.markdown(
    f"""
    <div class="masthead">
      <div class="brand">
        <div class="logo">{brand_mark(46, "main")}</div>
        <div>
          <div class="eyebrow">{eyebrow}</div>
          <h1 class="wordmark">{heading}</h1>
          <div class="mastline">{mastline}</div>
        </div>
      </div>
      <div class="chips">{chips}</div>
    </div>
    <div class="lede">{lede}</div>
    """,
    unsafe_allow_html=True,
)


x = pd.DataFrame([cleaned])[FEATURES]

try:
    risk = float(model.predict_proba(x)[0, 1])
except Exception as exc:
    if st.session_state.get("_retrain_attempts", 0) >= 1:
        st.error("❌ The saved model could not score these inputs, even after retraining.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        st.stop()

    st.session_state["_retrain_attempts"] = st.session_state.get("_retrain_attempts", 0) + 1
    st.warning("The saved model didn't match these inputs (`"
               f"{type(exc).__name__}`). Retraining automatically…")
    with st.spinner("Retraining the model — about 60 seconds…"):
        try:
            run_training()
        except TrainingError as train_exc:
            st.error(f"❌ Automatic retraining failed.\n\n{train_exc}")
            st.stop()
    st.cache_resource.clear()
    st.rerun()
else:
    st.session_state["_retrain_attempts"] = 0

risk = min(max(risk, 0.0), 1.0)
risk_interval = prediction_interval(x, risk)

if missing_features:
    st.caption(
        f"This prediction uses **{len(FEATURES) - len(missing_features)} of "
        f"{len(FEATURES)}** measured values — "
        f"{', '.join(missing_features)} were imputed.")

# ── Patient view: a deliberately different surface, not a re-skin ──────────
if mode == "Patient":
    patient_recs, patient_combined = improvement_scenarios(cleaned, model)
    render_patient_view(
        risk=risk, cleaned=cleaned, missing_features=missing_features,
        interval=risk_interval, flags=get_flags(cleaned),
        rec_rows=patient_recs, combined_risk=patient_combined,
        drivers=shap_drivers(x))
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab_hist, tab_batch, tab_report = st.tabs([
    "Risk",
    "Explain",
    "Performance",
    "Data",
    "Compare",
    "History",
    "Batch",
    "Report",
])


# ── Tab 1: Prediction ─────────────────────────────────────────────────────
with tab1:
    level = "Low" if risk < .3 else "Moderate" if risk < .6 else "High"

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Estimated risk")
            gauge_fig = risk_gauge(risk, THRESHOLD, level)
            st.pyplot(gauge_fig)
            plt.close(gauge_fig)
            if risk_interval:
                st.caption(f"90% interval: **{risk_interval[0]:.1%} – {risk_interval[1]:.1%}** "
                           f"· {risk_interval[2]} similar patients")
            st.caption(
                "Bands: 0–30% low · 30–60% moderate · 60–100% high. "
                f"The dark line marks the screening threshold ({THRESHOLD:.2f}).")

    with right:
        with st.container(border=True):
            st.metric("Risk level", level)
            st.markdown(
                f'<div class="bandline" style="color:{BAND_COLOUR[level]}">'
                f'<span class="banddot" style="background:{BAND_COLOUR[level]}"></span>'
                f'{level} risk band</div>'
                f'<div class="bandcap">Screening threshold {THRESHOLD:.2f}</div>',
                unsafe_allow_html=True)

            if risk >= .6:
                st.error(
                    "**High Risk Detected.** This patient shows strong indicators of diabetes. "
                    "Immediate consultation with a healthcare professional is strongly recommended. "
                    "Further diagnostic tests such as fasting blood glucose and HbA1c are advised.")
            elif risk >= .3:
                st.warning(
                    "**Moderate Risk Detected.** Some risk factors are present. "
                    "Lifestyle modifications such as diet improvement and increased physical activity "
                    "are recommended. A screening test would be advisable.")
            else:
                st.success(
                    "**Low Risk.** The patient's current values suggest a low likelihood of diabetes. "
                    "Maintaining a healthy lifestyle and regular checkups is still recommended.")

    st.markdown("---")
    d1, d2 = st.columns(2, gap="large")

    with d1:
        with st.container(border=True):
            st.markdown("#### Screening decision")
            if risk >= THRESHOLD:
                st.error(
                    f"**Positive at the tuned threshold ({THRESHOLD:.2f}).** "
                    "Confirm with fasting plasma glucose or HbA1c before any diagnosis.")
            else:
                st.success(
                    f"**Negative at the tuned threshold ({THRESHOLD:.2f}).** "
                    "Routine screening still applies if other risk factors are present.")
            if AT_THRESHOLD and AT_DEFAULT:
                st.caption(
                    f"Cut-off rule: {THRESHOLD_RULE}. On the held-out test set, recall is "
                    f"**{AT_THRESHOLD['recall']:.0%}** at this threshold versus "
                    f"{AT_DEFAULT['recall']:.0%} at the default 0.50.")

    with d2:
        with st.container(border=True):
            st.markdown("#### How certain is this number?")
            if risk_interval:
                low, high, neighbours = risk_interval
                st.markdown(
                    f"The model predicts **{risk:.1%}**, with a 90% interval of "
                    f"**{low:.1%} – {high:.1%}**.")
                cohort_note = similar_patients_summary(x, k=neighbours)
                if cohort_note:
                    st.markdown(cohort_note)
                st.caption(
                    f"The width comes from how widely risk varies among the {neighbours} training "
                    "patients most similar to this one: a tight cluster gives a narrow interval, a "
                    "mixed one gives a wide interval. It describes uncertainty for comparable "
                    "patients, not measurement error.")
                if low < THRESHOLD <= high:
                    st.warning("Borderline case: part of the interval lies above the screening "
                               "threshold, so the positive/negative call is not clear-cut.")
            else:
                st.caption("Uncertainty estimate unavailable for this model.")

    # ── Before / after: improving one modifiable value ──────────────────────
    st.markdown("---")
    st.subheader("Before / After: improving one risk factor")
    st.markdown(
        "This re-scores the patient with **one** modifiable measurement moved to a healthy "
        "target, leaving everything else unchanged. It is a what-if scenario built from the "
        "model's learned relationships — not a forecast, because real improvement also depends "
        "on factors the model never sees.")

    with st.container(border=True):
        next_risk = None
        ctl, info = st.columns([1.3, 1.7], gap="large")

        with ctl:
            chosen = st.selectbox(
                "Value to improve",
                list(MODIFIABLE),
                format_func=lambda f: FEATURE_INFO[f]["label"],
                key="improve_feature",
            )
        target, unit, target_note = MODIFIABLE[chosen]
        current_value = cleaned[chosen]

        with info:
            if pd.isna(current_value):
                st.info(f"**{FEATURE_INFO[chosen]['label']}** was not measured, so there is "
                        "nothing to compare. Enter a reading in the sidebar to model this.")
            elif float(current_value) <= float(target):
                st.success(f"**{FEATURE_INFO[chosen]['label']}** is already at or below the "
                           f"healthy target ({target:g} {unit}), so there is no improvement "
                           "left to model. Try another value.")
            else:
                try:
                    improved = cleaned.copy()
                    improved[chosen] = float(target)
                    next_risk = float(
                        model.predict_proba(pd.DataFrame([improved])[FEATURES])[0, 1])
                    next_risk = min(max(next_risk, 0.0), 1.0)
                except Exception as exc:  # noqa: BLE001
                    st.warning("The comparison could not be scored for this value. "
                               f"({type(exc).__name__})")

                if next_risk is not None:
                    units = FEATURE_INFO[chosen].get("unit") or unit
                    st.markdown(
                        f"**{FEATURE_INFO[chosen]['label']}**: "
                        f"{fmt_value(chosen, current_value)} → **{target:g}** {units}  \n"
                        f"<span style='color:#57606a'>{target_note}.</span>",
                        unsafe_allow_html=True)

        if next_risk is not None:
            delta = next_risk - risk
            e1, e2, e3 = st.columns(3)
            e1.metric("Current risk", f"{risk:.1%}")
            e2.metric("After improvement", f"{next_risk:.1%}",
                      delta=f"{delta * 100:+.1f} pts",
                      delta_color="inverse")
            e3.metric("Relative change",
                      f"{delta / risk:.0%}" if risk > 1e-6 else "n/a")



            if next_risk < THRESHOLD <= risk:
                st.success(
                    f"This single change would move the patient below the screening "
                    f"threshold ({THRESHOLD:.2f}) — from a positive to a negative call.")
            elif delta < 0:
                st.caption(
                    f"That one change lowers the estimated risk by "
                    f"**{abs(delta) * 100:.1f} percentage points**. Changing more than "
                    "one factor usually compounds the benefit.")
        else:
            st.caption("The comparison appears once a value above its healthy target is "
                       "selected.")

    # ── Stage 4: recommendation engine ─────────────────────────────────────
    st.markdown("---")
    st.subheader("What to improve first")
    st.markdown(
        "Each modifiable value is moved to its healthy target **one at a time**, then "
        "the scenarios are ranked by the risk that remains. The top row is the single "
        "change with the largest modelled benefit for this patient.")
    rec_rows, combined_risk = improvement_scenarios(cleaned, model)

    with st.container(border=True):
        if not rec_rows:
            st.success(
                "Every modifiable value is already at or below its healthy target, so "
                "there is no single-change scenario left to rank. Maintaining these "
                "values is the recommendation.")
        else:
            best = rec_rows[0]
            best_gain = (risk - best["new_risk"]) * 100
            total_gain = ((risk - combined_risk) * 100
                          if combined_risk is not None else None)

            m1, m2, m3 = st.columns(3)
            m1.metric("Change this first", COMPACT_LABELS[best["feature"]],
                      delta=f"-{best_gain:.1f} pts" if best_gain > 0 else None,
                      delta_color="inverse" if best_gain > 0 else "off",
                      help=f"Move {short_label(best['feature'])} from "
                           f"{fmt_value(best['feature'], best['current'])} to "
                           f"{best['target']:g} {best['unit']}.")
            m2.metric("Risk after that change", f"{best['new_risk']:.1%}")
            m3.metric("If every target is reached",
                      f"{combined_risk:.1%}" if combined_risk is not None else "n/a",
                      delta=(f"-{total_gain:.1f} pts"
                             if total_gain is not None and total_gain > 0 else None),
                      delta_color="inverse" if total_gain is not None and total_gain > 0
                      else "off")

            rec_df = pd.DataFrame({
                "Priority":       [row["rank"] for row in rec_rows],
                "Feature":        [short_label(row["feature"]) for row in rec_rows],
                "Current":        [fmt_value(row["feature"], row["current"])
                                   for row in rec_rows],
                "Healthy target": [f"{row['target']:g} {row['unit']}" for row in rec_rows],
                "Risk after":     [f"{row['new_risk']:.1%}" for row in rec_rows],
                "Reduction":      [f"−{(risk - row['new_risk']) * 100:.1f} pts"
                                   for row in rec_rows],
                "Typical effort": [EFFORT[row["feature"]] for row in rec_rows],
            })
            st.dataframe(rec_df, hide_index=True, use_container_width=True)

            if best_gain > 0:
                st.markdown(
                    f"**Start with {short_label(best['feature'])}.** Bringing it from "
                    f"{fmt_value(best['feature'], best['current'])} to "
                    f"**{best['target']:g} {best['unit']}** is the single largest modelled "
                    f"reduction ({best_gain:.1f} points).")
            if combined_risk is not None:
                st.caption(
                    f"Reaching every target together is modelled at **{combined_risk:.1%}**"
                    + (f" ({total_gain:.1f} points lower)." if total_gain is not None else ".")
                    + " Treat that as an optimistic ceiling: the model moves one value at a "
                    "time, while in reality diet, weight, glucose and insulin change together "
                    "and interact.")
                if combined_risk < THRESHOLD <= risk:
                    st.success(
                        f"Combined, these changes would move the patient from a positive to "
                        f"a negative screening call at the {THRESHOLD:.2f} threshold.")

    if missing_features:
        st.info(
            "ℹ️ **Missing measurements were imputed.** "
            + ", ".join(FEATURE_INFO[f]["label"] for f in missing_features)
            + " had no reading, so the model used the dataset median. "
            "Entering real measurements gives a more trustworthy result.")

    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Clinical flags")
        flags = get_flags(cleaned)
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("No clinical thresholds exceeded. All entered values appear "
                       "within acceptable ranges.")

    with st.container(border=True):
        st.markdown("#### Patient summary")
        summary = pd.DataFrame({
            "Feature":      [FEATURE_INFO[f]["label"] for f in FEATURES],
            "Your Value":   [fmt_value(f, cleaned[f]) for f in FEATURES],
            "Normal Range": [FEATURE_INFO[f]["normal"] for f in FEATURES],
            "Status":       [field_status(f, cleaned[f]) for f in FEATURES],
        })

        def _tint_status(value):
            text, tint = STATUS_STYLE.get(value, ("#1f2933", "#ffffff"))
            return f"background-color: {tint}; color: {text}; font-weight: 600"

        st.dataframe(summary.style.map(_tint_status, subset=["Status"]),
                     hide_index=True, use_container_width=True)

    st.markdown("---")
    save_prediction_panel(risk, cleaned, missing_features, "clin")

# ── Tab 2: Explain (Detailed clinical narrative, zero charts) ──────────────
with tab2:
    explanation_data = build_explanation(x, cleaned, risk)

    st.subheader(f"Why this result — {explanation_data['headline']}")
    level = risk_level(risk)
    st.markdown(
        f'<div class="bandline" style="font-size:1.15rem;margin-bottom:8px;color:{BAND_COLOUR[level]}">'
        f'<span class="banddot" style="background:{BAND_COLOUR[level]}"></span>'
        f'{level} risk · {explanation_data["headline"]}</div>',
        unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(synthesize_clinical_narrative(cleaned, risk))
        cohort_line = similar_patients_summary(x)
        if cohort_line:
            st.caption(cohort_line)

    col_raises, col_lowers = st.columns(2, gap="large")
    with col_raises:
        st.markdown("### What raises this result")
        if not explanation_data["raises"]:
            st.info("No single feature is strongly pushing risk upward.")
        else:
            for card in explanation_data["raises"][:4]:
                with st.container(border=True):
                    text_col, tint_col = STATUS_STYLE.get(card["status"], ("#344054", "#f0f0f0"))
                    st.markdown(
                        f"**{card['label']}** · `{card['value']}` "
                        f"<span style='font-size:.72rem;background:{tint_col};color:{text_col};"
                        f"padding:2px 8px;border-radius:999px;font-weight:600;'>{card['status']}</span> "
                        f"<span style='font-size:.75rem;color:#667085;'>· {card['strength']}</span>",
                        unsafe_allow_html=True)
                    if card.get("imputed_note"):
                        st.caption(card["imputed_note"])
                    st.markdown(card["sentence"])

    with col_lowers:
        st.markdown("### What lowers it")
        if not explanation_data["lowers"]:
            st.info("No feature is clearly reducing risk here.")
        else:
            for card in explanation_data["lowers"][:4]:
                with st.container(border=True):
                    text_col, tint_col = STATUS_STYLE.get(card["status"], ("#344054", "#f0f0f0"))
                    st.markdown(
                        f"**{card['label']}** · `{card['value']}` "
                        f"<span style='font-size:.72rem;background:{tint_col};color:{text_col};"
                        f"padding:2px 8px;border-radius:999px;font-weight:600;'>{card['status']}</span> "
                        f"<span style='font-size:.75rem;color:#667085;'>· {card['strength']}</span>",
                        unsafe_allow_html=True)
                    if card.get("imputed_note"):
                        st.caption(card["imputed_note"])
                    st.markdown(card["sentence"])

    st.markdown("---")
    st.markdown("### Where this patient sits")
    st.caption("Green shows the normal reference band. The dark marker shows this patient's reading.")
    range_cols = st.columns(min(4, len(FEATURES)))
    for idx, feat in enumerate(FEATURES):
        rb = explanation_data["range_bars"][feat]
        current_val = rb["current"]
        low_b, high_b = float(rb["normal_low"]), float(rb["normal_high"])
        max_display = max(high_b * 1.5, (current_val * 1.15) if current_val is not None else high_b * 1.5, 1.0)
        low_pct = min(100.0, max(0.0, (low_b / max_display) * 100.0))
        high_pct = min(100.0, max(low_pct, (high_b / max_display) * 100.0))

        with range_cols[idx % 4]:
            with st.container(border=True):
                st.caption(f"**{rb['label']}**")
                if current_val is None:
                    bar_bg = (
                        '<div style="position:relative;height:18px;background:#eef1f4;'
                        'border-radius:8px;overflow:hidden;margin-top:4px;"></div>'
                    )
                else:
                    left_pct = min(98.0, max(2.0, (current_val / max_display) * 100.0))
                    bar_bg = (
                        f'<div style="position:relative;height:18px;background:#eef1f4;'
                        f'border-radius:8px;overflow:hidden;margin-top:4px;">'
                        f'<div style="position:absolute;left:0;top:0;width:100%;height:100%;'
                        f'background:linear-gradient(90deg, #fff4d6 0%, #fff4d6 {low_pct:.1f}%, '
                        f'#e6f4ea {low_pct:.1f}%, #e6f4ea {high_pct:.1f}%, '
                        f'#fde8e6 {high_pct:.1f}%, #fde8e6 100%);border-radius:8px;"></div>'
                        f'<div style="position:absolute;left:{left_pct:.1f}%;top:3px;'
                        f'width:4px;height:12px;background:#101828;border-radius:2px;"></div>'
                        f'</div>'
                    )
                st.markdown(bar_bg, unsafe_allow_html=True)
                st.caption(f"**{rb['value']}** · {FEATURE_INFO[feat]['normal']}")

    st.markdown("---")
    st.markdown("### Try a change")
    try_change_feature = st.selectbox(
        "Which value to adjust?", list(MODIFIABLE.keys()),
        format_func=lambda f: FEATURE_INFO[f]["label"], key="explain_modify")
    target_val, target_unit, target_note = MODIFIABLE[try_change_feature]
    current_for_change = cleaned.get(try_change_feature)
    slider_key = f"try_change_{try_change_feature}"

    has_current = current_for_change is not None and not (isinstance(current_for_change, float) and np.isnan(current_for_change))
    base_val = float(current_for_change) if has_current else float(target_val)
    low_bound = max(1.0, min(base_val, float(target_val)) - 25.0)
    high_bound = max(base_val, float(target_val)) + 30.0
    step_val = 0.1 if try_change_feature == "BMI" else 1.0

    adjusted = st.slider(
        f"Move {FEATURE_INFO[try_change_feature]['label']} to",
        min_value=float(low_bound),
        max_value=float(high_bound),
        value=float(base_val),
        step=float(step_val),
        key=slider_key)

    try:
        changed = cleaned.copy()
        changed[try_change_feature] = float(adjusted)
        new_risk_change = float(model.predict_proba(pd.DataFrame([changed])[FEATURES])[0, 1])
        new_risk_change = min(max(new_risk_change, 0.0), 1.0)
        sentence = (f"Changing **{FEATURE_INFO[try_change_feature]['label']}** from "
                    f"**{fmt_value(try_change_feature, current_for_change) if has_current else 'not measured'}** "
                    f"to **{adjusted:g} {target_unit}** moves the estimate from **{risk:.0%}** to **{new_risk_change:.0%}** "
                    f"({(new_risk_change - risk) * 100:+.1f} percentage points).")
    except Exception as exc:
        sentence = f"Could not re-score this change ({type(exc).__name__})."
    st.info(sentence)

    st.markdown("---")
    with st.container(border=True):
        st.markdown("### How sure is this?")
        if risk_interval:
            low, high, neighbours = risk_interval
            st.markdown(
                f"People with very similar measurements scored between **{low:.0%}** and **{high:.0%}**. "
                f"That range ({neighbours} comparable patients) reflects how spread out the results are.")
            if mode != "Patient" and low < THRESHOLD <= high:
                st.warning("Borderline case: part of the interval straddles the screening threshold, so the positive/negative call is not clear-cut.")
        else:
            st.caption("An uncertainty range is not available for this profile.")
        st.markdown(f"**{explanation_data['warning_text']}**")
        if mode != "Patient":
            st.caption(f"Screening threshold is **{THRESHOLD:.2f}**. Positive if risk is at or above it; negative below it.")

    if mode == "Patient":
        with st.container(border=True):
            st.markdown("### Worth asking your doctor about")
            st.markdown(
                "This is a screening estimate, not a diagnosis. If anything in the result worries you — "
                "especially missing measurements or values marked abnormal — it is worth discussing with a clinician.")
            if explanation_data["missing_features"]:
                st.caption("Some inputs were estimated rather than measured. A real reading would make the result more reliable.")
# ── Tab 3: Model Performance ──────────────────────────────────────────────
with tab3:
    st.subheader("Model comparison")
    st.markdown(
        "Four models were trained and compared using **5-fold stratified cross-validation**: "
        "Logistic Regression, Random Forest, Gradient Boosting and XGBoost. "
        "Class imbalance (only ~35% diabetic cases) is handled with **balanced class "
        "weights** — `class_weight=\"balanced\"`, and `scale_pos_weight` for XGBoost — "
        "**not SMOTE**, which keeps the saved model self-contained and deployment-safe. "
        "The best model was selected based on **AUC-ROC score**."
    )

    results = meta.get("results") or {}
    if not results:
        st.warning("No model metrics were found in `metrics.json`. Retrain the model to "
                   "regenerate them.")
    else:
        res = pd.DataFrame(results).T.drop(columns="params", errors="ignore")
        st.dataframe(
            res.style.highlight_max(axis=0, color="#cfe8cf").format("{:.4f}"),
            use_container_width=True)

        best_params = (results.get(meta.get("best_model", ""), {}) or {}).get("params")
        if best_params:
            with st.expander("Best hyper-parameters"):
                st.json(best_params)

        # ── Threshold tuning ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Decision threshold tuning")
        st.markdown(
            "A 0.50 cut-off is arbitrary. Because missing a diabetic patient costs more than "
            "a false alarm, the threshold was tuned on **out-of-fold** predictions to "
            "maximise recall while keeping precision at or above 50%."
        )
        st.info(f"**Selected threshold: {THRESHOLD:.2f}** — {THRESHOLD_RULE}.")

        if AT_THRESHOLD and AT_DEFAULT:
            compare = pd.DataFrame({
                "Operating point": ["Default (0.50)", f"Tuned ({THRESHOLD:.2f})"],
                "Accuracy":  [AT_DEFAULT["accuracy"],  AT_THRESHOLD["accuracy"]],
                "Precision": [AT_DEFAULT["precision"], AT_THRESHOLD["precision"]],
                "Recall":    [AT_DEFAULT["recall"],    AT_THRESHOLD["recall"]],
                "F1":        [AT_DEFAULT["f1"],        AT_THRESHOLD["f1"]],
            })
            st.dataframe(
                compare.style.highlight_max(subset=["Recall"], color="#cfe8cf")
                       .format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1"]}),
                hide_index=True, use_container_width=True)

        sweep = meta.get("threshold_sweep") or []
        if sweep:
            sdf = pd.DataFrame(sweep).sort_values("threshold")
            figt, axt = plt.subplots(figsize=(8, 3.2))
            axt.plot(sdf["threshold"], sdf["recall"], color="tab:red", label="Recall")
            axt.plot(sdf["threshold"], sdf["precision"], color="tab:blue", label="Precision")
            axt.axvline(THRESHOLD, color="black", ls="--",
                        label=f"Chosen {THRESHOLD:.2f}")
            axt.set_xlabel("Decision threshold")
            axt.set_ylabel("Score")
            axt.set_ylim(0, 1.02)
            axt.set_title("Recall vs precision across thresholds (out-of-fold)")
            axt.legend()
            st.pyplot(figt)
            plt.close(figt)

        st.markdown("---")
        a, b = st.columns(2)

        with a:
            st.markdown("#### ROC curve")
            st.markdown(
                "The ROC curve shows the trade-off between catching true diabetics (recall) "
                "and avoiding false alarms. A curve closer to the top-left corner is better. "
                f"The best model (**{meta.get('best_model', 'unknown')}**) achieves the AUC shown below."
            )
            roc = meta.get("roc") or {}
            if roc.get("fpr") and roc.get("tpr"):
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.plot(roc["fpr"], roc["tpr"],
                        label=f"{meta.get('best_model', 'model')}", linewidth=2)
                ax.plot([0, 1], [0, 1], "--", c="grey", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ROC data is not available in `metrics.json`.")

        with b:
            st.markdown("#### Confusion matrix")
            st.markdown(
                "Shows how many patients were correctly and incorrectly classified **at the "
                f"tuned threshold of {THRESHOLD:.2f}**. "
                "**Top-left**: correctly identified as non-diabetic. "
                "**Bottom-right**: correctly identified as diabetic. "
                "**Bottom-left**: missed diabetics (false negatives) — the most costly errors in healthcare."
            )
            try:
                cm = np.array(meta["confusion"], dtype=int)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(cm, cmap="Blues")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, v, ha="center", va="center", fontsize=16,
                            fontweight="bold",
                            color="white" if v > cm.max() / 2 else "black")
                ax.set_xticks([0, 1], ["No Diabetes", "Diabetes"])
                ax.set_yticks([0, 1], ["No Diabetes", "Diabetes"])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                st.info("Confusion-matrix data is not available in `metrics.json`.")

        st.warning(
            "**Important note on recall:** In medical screening, missing a diabetic patient "
            "(false negative) is more dangerous than a false alarm (false positive). "
            f"That is why the cut-off was lowered from 0.50 to {THRESHOLD:.2f}: recall rises to "
            f"{AT_THRESHOLD.get('recall', float('nan')):.0%} at the cost of a few extra false "
            "alarms, which is the right trade-off for a screening tool."
        )

# ── Tab 4: Data Explorer ──────────────────────────────────────────────────
with tab4:
    st.subheader("Dataset explorer")

    explorer_error = None
    d = None
    try:
        d = load_data(DATA_PATH)
    except FileNotFoundError:
        explorer_error = (f"`diabetes.csv` was not found in `{BASE_DIR}`. "
                          "Add the Pima dataset file next to `app.py` to enable this tab.")
    except Exception as exc:  # noqa: BLE001
        explorer_error = f"`diabetes.csv` could not be read — {type(exc).__name__}: {exc}"

    if explorer_error:
        st.error("" + explorer_error)
        st.caption("The other three tabs are unaffected.")
    else:
        target_col = "Outcome" if "Outcome" in d.columns else "diabetes"

        if target_col not in d.columns:
            st.error("The dataset has no `Outcome` (or `diabetes`) column, so the "
                     "explorer cannot split patients by diabetes status.")
        else:
            st.markdown(
                f"The model was trained on **{len(d)} patient records** from the "
                f"Pima Indians Diabetes Database. **{d[target_col].mean():.1%}** of patients "
                "are diabetic. Use this tab to explore how features are distributed "
                "between diabetic and non-diabetic patients."
            )

            st.markdown("---")
            feat = st.selectbox("Select a feature to explore", FEATURES)
            st.markdown(f"**{FEATURE_INFO[feat]['help']}**")

            # 0 is a real value for Pregnancies, but a sentinel for the lab
            # measurements — only blank those out for the histogram.
            plot_df = d[[feat, target_col]]
            if feat in ZERO_AS_MISSING:
                plot_df = plot_df.replace({feat: {0: np.nan}})
            fig, ax = plt.subplots(figsize=(8, 4))
            for outcome, colour, label in [(0, "tab:blue", "No Diabetes"),
                                           (1, "tab:red", "Diabetes")]:
                ax.hist(plot_df.loc[plot_df[target_col] == outcome, feat].dropna(),
                        bins=30, alpha=0.6, color=colour, label=label)
            if pd.notna(cleaned[feat]):
                ax.axvline(cleaned[feat], c="black", ls="--", linewidth=2,
                           label=f"Current patient: {cleaned[feat]:g}")
            ax.set_xlabel(FEATURE_INFO[feat]["label"])
            ax.set_ylabel("Number of patients")
            ax.set_title(f"Distribution of {feat} by Diabetes Status")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            if feat in missing_features:
                st.caption(f"ℹ️ This patient's {feat} was not measured, so no reference "
                           "line is drawn.")

            st.markdown("---")
            st.markdown("#### Correlation heatmap")
            st.markdown(
                "Shows how strongly each pair of features is related. "
                "Values close to 1 or -1 mean strong correlation. "
                "Glucose and BMI tend to have the strongest correlation with diabetes outcome."
            )
            st.dataframe(d.corr(numeric_only=True).round(2), use_container_width=True)

# ── Tab 5: Patient comparison ─────────────────────────────────────────────
with tab5:
    st.subheader("Patient comparison")
    st.markdown(
        "Score a second patient profile against the one in the sidebar. **Patient A** is "
        "the profile you have been editing; **Patient B** is entered independently here. "
        "Use it to compare two people, or to see how far a patient would have to move to "
        "reach someone else's risk profile.")

    compare_on = st.toggle(
        "Enable comparison mode", value=False, key="compare_mode",
        help="Patient B gets its own eight measurements. Nothing here changes Patient A.")

    if not compare_on:
        st.info("Comparison mode is off. Turn it on to enter a second set of "
                "measurements and see both patients side by side.")
    else:
        # Start Patient B as a healthier version of Patient A: every modifiable
        # value that is above its healthy target is pre-filled at the target, so
        # the comparison is immediately meaningful instead of two identical rows.
        b_defaults = {}
        for feature in FEATURES:
            current = cleaned.get(feature)
            if current is None or pd.isna(current):
                current = MEDIANS.get(feature, 0.0)
            current = float(current)
            if feature in MODIFIABLE and current > float(MODIFIABLE[feature][0]):
                current = float(MODIFIABLE[feature][0])
            b_defaults[feature] = current

        comparison_specs = {
            "Pregnancies":              {"min_value": 0, "max_value": 20,
                                         "step": 1, "format": "%d"},
            "Glucose":                  {"min_value": 0, "max_value": 250,
                                         "step": 1, "format": "%d"},
            "BloodPressure":            {"min_value": 0, "max_value": 130,
                                         "step": 1, "format": "%d"},
            "SkinThickness":            {"min_value": 0, "max_value": 100,
                                         "step": 1, "format": "%d"},
            "Insulin":                  {"min_value": 0, "max_value": 850,
                                         "step": 1, "format": "%d"},
            "BMI":                      {"min_value": 0.0, "max_value": 70.0,
                                         "step": 0.1, "format": "%.1f"},
            "DiabetesPedigreeFunction": {"min_value": 0.05, "max_value": 2.5,
                                         "step": 0.01, "format": "%.2f"},
            "Age":                      {"min_value": 18, "max_value": 90,
                                         "step": 1, "format": "%d"},
        }

        st.markdown("#### Patient B measurements")
        st.caption("Patient B starts as Patient A with every modifiable value already at "
                   "its healthy target. Change any field to compare a different profile.")
        b_vals = {}
        field_cols = st.columns(4, gap="medium")
        for position, feature in enumerate(FEATURES):
            spec = comparison_specs[feature]
            value = b_defaults[feature]
            if spec["format"] == "%d":
                value = int(round(value))
            with field_cols[position % 4]:
                b_vals[feature] = st.number_input(
                    short_label(feature), value=value, key=f"cmp_{feature}",
                    help=FEATURE_INFO[feature]["help"], **spec)

        b_cleaned, b_issues, b_missing = validate_inputs(b_vals, MEDIANS)

        st.markdown("---")
        risk_b = None
        try:
            xb = pd.DataFrame([b_cleaned])[FEATURES]
            risk_b = min(max(float(model.predict_proba(xb)[0, 1]), 0.0), 1.0)
        except Exception as exc:  # noqa: BLE001
            st.error("❌ Patient B could not be scored. Patient A's result is unaffected.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

        if risk_b is not None:
            level_a = risk_level(risk)
            level_b = risk_level(risk_b)

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                with st.container(border=True):
                    st.markdown(f"#### Patient A · {level_a} risk")
                    gauge_a = risk_gauge(risk, THRESHOLD, level_a)
                    st.pyplot(gauge_a)
                    plt.close(gauge_a)
                    if risk >= THRESHOLD:
                        st.error(f"Positive at the {THRESHOLD:.2f} threshold.")
                    else:
                        st.success(f"Negative at the {THRESHOLD:.2f} threshold.")
                    if risk_interval:
                        st.caption(f"90% interval: {risk_interval[0]:.1%} – "
                                   f"{risk_interval[1]:.1%}")

            with col_b:
                with st.container(border=True):
                    st.markdown(f"#### Patient B · {level_b} risk")
                    gauge_b = risk_gauge(risk_b, THRESHOLD, level_b)
                    st.pyplot(gauge_b)
                    plt.close(gauge_b)
                    if risk_b >= THRESHOLD:
                        st.error(f"Positive at the {THRESHOLD:.2f} threshold.")
                    else:
                        st.success(f"Negative at the {THRESHOLD:.2f} threshold.")

            gap = risk_b - risk
            st.markdown("---")
            gap_col, verdict_col = st.columns([1, 2.2], gap="large")
            gap_col.metric("Difference (B − A)", f"{gap * 100:+.1f} pts")
            gap_col.caption(f"Patient A: **{risk:.1%}** · Patient B: **{risk_b:.1%}**")
            with verdict_col:
                if abs(gap) < 0.005:
                    st.info("The two profiles score almost identically.")
                elif gap > 0:
                    st.warning(
                        f"Patient B is **{gap * 100:.1f} points higher** than Patient A.")
                else:
                    st.success(
                        f"Patient B is **{abs(gap) * 100:.1f} points lower** than Patient A.")
                if (risk >= THRESHOLD) != (risk_b >= THRESHOLD):
                    lower_name = "B" if risk_b < risk else "A"
                    st.caption(
                        f"The screening call differs: Patient {lower_name} is the one "
                        f"below the {THRESHOLD:.2f} threshold.")
                if b_issues:
                    st.caption("Patient B has input warnings: "
                               + "; ".join(message for _, message in b_issues))
                if b_missing:
                    st.caption("Patient B imputed: "
                               + ", ".join(FEATURE_INFO[f]["label"] for f in b_missing))

            st.markdown("#### Feature-by-feature difference")
            diff_rows = []
            for feature in FEATURES:
                value_a = cleaned.get(feature)
                value_b = b_cleaned.get(feature)
                if value_a is None or pd.isna(value_a):
                    value_a = MEDIANS.get(feature)
                if value_b is None or pd.isna(value_b):
                    value_b = MEDIANS.get(feature)
                try:
                    delta_value = float(value_b) - float(value_a)
                    delta_text = f"{delta_value:+g}"
                except (TypeError, ValueError):
                    delta_text = "n/a"
                diff_rows.append({
                    "Feature":    FEATURE_INFO[feature]["label"],
                    "Patient A":  fmt_value(feature, value_a),
                    "Patient B":  fmt_value(feature, value_b),
                    "B − A":      delta_text,
                    "A status":   field_status(feature, value_a),
                    "B status":   field_status(feature, value_b),
                })
            diff_df = pd.DataFrame(diff_rows)
            st.dataframe(
                diff_df.style.map(tint_status, subset=["A status", "B status"]),
                hide_index=True, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Top drivers for each patient")
            if explainer is None:
                st.caption("The explainability engine is unavailable, so the drivers cannot be listed. "
                           "The scores and comparison above are unaffected.")
            else:
                driver_cols = st.columns(2, gap="large")
                for column, name, frame in (
                        (driver_cols[0], "Patient A", x),
                        (driver_cols[1], "Patient B", xb)):
                    with column:
                        st.markdown(f"**{name}**")
                        try:
                            values = explainer(frame)
                            vector = np.asarray(values.values)[0]
                            total_abs = float(np.sum(np.abs(vector))) + 1e-9
                            order = np.argsort(-np.abs(vector))[:3]
                            for idx in order:
                                direction = "raises" if vector[idx] > 0 else "lowers"
                                share_pct = abs(float(vector[idx])) / total_abs * 100.0
                                st.markdown(
                                    f"- **{short_label(FEATURES[idx])}**: "
                                    f"{direction} risk ({share_pct:.0f}% of decision)")
                        except Exception as exc:  # noqa: BLE001
                            st.caption(f"Drivers unavailable ({type(exc).__name__}).")

            st.caption(
                "As everywhere else in this app, the scores are model estimates from eight "
                "routine measurements, not diagnoses. A difference between two profiles "
                "shows how the model responds to their inputs, not a treatment effect.")

# ── Tab 6: Patient history ────────────────────────────────────────────────
with tab_hist:
    st.subheader("Patient history")
    st.markdown(
        "Every prediction you save is kept **in this app session's memory** "
        "(never written to a database or file) and plotted over time. Signing "
        "out or closing the session clears it, so use the downloads to keep a "
        "permanent copy or move records to another session.")
    render_history("clin_hist")

# ── Tab 7: Batch prediction ───────────────────────────────────────────────
with tab_batch:
    st.subheader("Batch prediction")
    st.markdown(
        "Upload a CSV with one row per patient and the app scores the whole "
        f"cohort in one pass (up to {BATCH_LIMIT:,} rows). Missing values and "
        "clinical zeros are handled the same way as in the single-patient form. "
        "Rows are scored independently — do not include an outcome column. "
        "The file stays in session memory and is never written to disk.")
    template = pd.DataFrame([{
        "Pregnancies": 1, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 25, "Insulin": 80, "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.4, "Age": 35,
    }])
    st.download_button(
        "⬇️ Download CSV template", template.to_csv(index=False).encode("utf-8"),
        "diabetes_batch_template.csv", "text/csv", key="batch_template")
    st.caption("Required columns: **" + ", ".join(FEATURES) + "**. A `0` in "
               "Glucose, BloodPressure, SkinThickness, Insulin or BMI means "
               "*not measured* and is imputed.")

    uploaded = st.file_uploader(
        "Upload patient CSV", type=["csv"], key="batch_upload",
        help="One patient per row. Extra columns are preserved in the download.")
    demo = st.checkbox(
        "Use 20 example patients from the training data instead",
        key="batch_demo")

    frame = None
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ That file could not be read as CSV "
                     f"(`{type(exc).__name__}`). Check the delimiter and header.")
    elif demo:
        try:
            frame = load_data(DATA_PATH).head(20).drop(
                columns=["Outcome"], errors="ignore")
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ The example data could not be loaded "
                     f"(`{type(exc).__name__}`).")

    if frame is None:
        st.info("Upload a CSV or tick the example box to see how batch scoring "
                "works.")
    else:
        try:
            results, probabilities, notes = batch_predict(frame, model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ The batch could not be scored "
                     f"(`{type(exc).__name__}`). The single-patient form is "
                     "unaffected.")
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
        else:
            for note in notes:
                st.warning(note)
            total = len(results)
            positives = int((probabilities >= THRESHOLD).sum())
            high_band = sum(risk_level(value) == "High" for value in probabilities)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Patients scored", f"{total:,}")
            m2.metric("Positive at threshold", f"{positives:,}")
            m3.metric("Mean risk", f"{probabilities.mean():.1%}")
            m4.metric("High band", f"{high_band:,}")

            if total >= 2:
                fig_batch, ax_batch = plt.subplots(figsize=(7, 2.6))
                bins = int(min(20, max(5, total // 2)))
                ax_batch.hist(probabilities * 100, bins=bins, color="#1f6feb",
                              alpha=0.75, edgecolor="white")
                ax_batch.axvline(THRESHOLD * 100, color="#111827", ls="--",
                                 lw=1.4)
                ax_batch.text(THRESHOLD * 100 + 1.5,
                              ax_batch.get_ylim()[1] * 0.92,
                              f"threshold {THRESHOLD:.2f}", fontsize=7,
                              color="#111827")
                ax_batch.set_xlabel("Predicted risk (%)", fontsize=9)
                ax_batch.set_ylabel("Patients", fontsize=9)
                ax_batch.set_xlim(0, 100)
                ax_batch.grid(axis="y", color="#eef1f4", lw=0.8)
                ax_batch.set_axisbelow(True)
                ax_batch.tick_params(labelsize=8)
                for side in ("top", "right"):
                    ax_batch.spines[side].set_visible(False)
                fig_batch.tight_layout()
                st.pyplot(fig_batch)
                plt.close(fig_batch)

            shown = results.head(200)
            st.dataframe(shown, hide_index=True, use_container_width=True)
            if total > len(shown):
                st.caption(f"Showing the first {len(shown)} of {total:,} rows. "
                           "The download contains every row.")
            st.download_button(
                "⬇️ Download results CSV",
                results.to_csv(index=False).encode("utf-8"),
                "batch_predictions.csv", "text/csv", key="batch_results_csv")

# ── Tab 8: PDF report ─────────────────────────────────────────────────────
with tab_report:
    st.subheader("PDF report")
    render_report_section(
        "clin_report", risk=risk, cleaned=cleaned, interval=risk_interval,
        flags=get_flags(cleaned), drivers=shap_drivers(x),
        default_label=st.session_state.get("last_label", ""))

st.divider()
st.caption(
    "Trained on the Pima Indians Diabetes Dataset (adult women of Pima heritage, aged 21+). "
    "Class imbalance handled with balanced class weights (no SMOTE). "
    "An explainability engine is used for driver analysis; the decision threshold is tuned for high recall. "
    "Saved predictions are kept in session memory only and reset on sign-out. "
    "This tool is for educational purposes only and is not a medical diagnosis."
)
