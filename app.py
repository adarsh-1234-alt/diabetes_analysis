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
* LIME runs next to SHAP on the same patient: two importance charts, a rank
  correlation and a sign-agreement count make it obvious when the two methods
  tell the same story and when they disagree.
* SHAP dependence plots for the three globally strongest features show whether a
  feature's effect is linear, saturating or non-monotone.
* A two-feature interaction heat map sweeps both features at once, marks where
  this patient sits and quantifies whether the pair behaves additively.
* Patient comparison scores a second profile side by side with the sidebar one.
* The recommendation engine ranks every modifiable value by how much risk it
  would remove if brought to its healthy target, plus the combined scenario.
"""
import hmac
import json
import os
import subprocess
import sys
import threading
import traceback

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

try:  # optional: only needed if the saved model is an XGBoost pipeline
    import xgboost
    XGBOOST_VERSION = xgboost.__version__
except Exception:  # noqa: BLE001
    XGBOOST_VERSION = None

try:  # Stage 4: a second, model-agnostic explanation method.
    from lime.lime_tabular import LimeTabularExplainer
    LIME_OK = True
except Exception:  # noqa: BLE001
    LimeTabularExplainer = None
    LIME_OK = False

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
SHAP_GLOBAL_TIMEOUT = 120

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
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


# ── Plain English explanations ───────────────────────────────────────────────
def explain_feature_up(feat, val):
    explanations = {
        "Glucose": (
            f"Your blood sugar is **{val:.0f} mg/dL** — "
            + ("this is very high and strongly indicates diabetes. Glucose is the most important predictor in this model."
               if val >= 200 else
               "this is in the elevated range. High blood sugar means your body is struggling to process glucose properly."
               if val >= 140 else
               "even at this level, blood sugar is pushing your risk upward slightly.")
        ),
        "BMI": (
            f"Your BMI is **{val:.1f}** — "
            + ("this is in the obese range. Excess body fat, especially around the abdomen, makes cells resistant to insulin."
               if val >= 30 else
               "this is in the overweight range, which increases the workload on your pancreas to produce insulin.")
        ),
        "Age": (
            f"At **{val:.0f} years old**, age is working against you here. "
            "As we age, insulin sensitivity naturally decreases and the pancreas becomes less efficient, "
            "which is why diabetes becomes more common with age."
        ),
        "Pregnancies": (
            f"Having had **{val:.0f} {'pregnancy' if val == 1 else 'pregnancies'}** is a contributing factor. "
            "Each pregnancy puts stress on the body's insulin production, and gestational diabetes "
            "during any pregnancy is a strong predictor of Type 2 diabetes later in life."
        ),
        "Insulin": (
            f"Your insulin level of **{val:.0f} mu U/ml** is raising a flag. "
            "Abnormal insulin levels — either too high or too low — suggest the body is struggling "
            "to regulate blood sugar properly, which is a hallmark of diabetes progression."
        ),
        "BloodPressure": (
            f"Your blood pressure of **{val:.0f} mm Hg** is on the higher side. "
            "High blood pressure is closely linked to insulin resistance — "
            "both often appear together and reinforce each other's damage."
        ),
        "SkinThickness": (
            f"A skin fold thickness of **{val:.0f} mm** suggests higher body fat levels. "
            "More body fat, particularly in certain areas, is associated with reduced insulin sensitivity "
            "and a higher chance of developing Type 2 diabetes."
        ),
        "DiabetesPedigreeFunction": (
            f"Your family history score is **{val:.2f}** — "
            + ("this is quite high. Having close relatives with diabetes significantly increases your genetic risk. "
               "Diabetes has a strong hereditary component."
               if val > 0.5 else
               "there is some family history of diabetes in your background, which adds a small but real genetic risk factor.")
        ),
    }
    return explanations.get(feat, f"**{feat}** (value: {val:.2f}) is contributing to higher risk.")


def explain_feature_down(feat, val):
    explanations = {
        "Glucose": (
            f"Your blood sugar of **{val:.0f} mg/dL** is within a healthy range. "
            "This is actually one of the strongest protective factors — "
            "your body appears to be processing glucose efficiently."
        ),
        "BMI": (
            f"Your BMI of **{val:.1f}** is in a {'healthy' if val < 25 else 'acceptable'} range. "
            "Maintaining a healthy weight is one of the most powerful protections against diabetes, "
            "as it keeps your cells sensitive to insulin."
        ),
        "Age": (
            f"Being **{val:.0f} years old** means age is not a significant risk factor here. "
            "Younger age is generally protective against Type 2 diabetes."
        ),
        "Pregnancies": (
            f"{'No pregnancies' if val == 0 else f'Having had {val:.0f} pregnancy/pregnancies'} "
            "is not adding significant hormonal risk in your case."
        ),
        "Insulin": (
            f"Your insulin level of **{val:.0f} mu U/ml** appears reasonable. "
            "This suggests your body is managing blood sugar regulation relatively well."
        ),
        "BloodPressure": (
            f"Your blood pressure of **{val:.0f} mm Hg** is in a healthy range. "
            "Good blood pressure control is closely linked to better insulin sensitivity."
        ),
        "SkinThickness": (
            f"Your skin fold thickness of **{val:.0f} mm** is not raising any red flags. "
            "This suggests body fat levels are not a major concern at this point."
        ),
        "DiabetesPedigreeFunction": (
            f"Your family history score of **{val:.2f}** is relatively low. "
            "Limited family history of diabetes means your genetic risk is reduced, "
            "which is working in your favour."
        ),
    }
    return explanations.get(feat, f"**{feat}** (value: {val:.2f}) is helping reduce your risk.")


def explain_missing(feat, median):
    filled = f"{median:.1f}" if median is not None and not pd.isna(median) else "the median"
    return (
        f"**{FEATURE_INFO[feat]['label']}** was not measured, so the model filled in "
        f"**{filled}** (the training median). Its contribution shown here is an estimate, "
        "not a personal measurement."
    )


# ── Auth ────────────────────────────────────────────────────────────────────
def login():
    if st.session_state.get("auth"):
        return True
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("Please sign in to continue.")
    with st.form("login"):
        u  = st.text_input("Username")
        p  = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in")
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
    return False


if not login():
    st.stop()


# ── Stage 3: presentation layer ─────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; }
      h1 { letter-spacing: -0.02em; font-size: 2.35rem; }
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
    </style>
    """,
    unsafe_allow_html=True,
)

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


def rank_correlation(a, b):
    """Spearman rank correlation of *absolute* importance.

    SHAP and LIME measure importance on different scales and against different
    reference points (dataset average vs local neighbourhood), so asking whether
    they rank the features' magnitudes the same way is the fair comparison. The
    sign of each weight is reported separately, feature by feature.
    """
    a = pd.Series(np.abs(np.asarray(a, dtype=float))).rank().to_numpy()
    b = pd.Series(np.abs(np.asarray(b, dtype=float))).rank().to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def render_contribution_bars(values, title):
    """One horizontal importance chart, red = raises risk, blue = lowers it."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))
    labels = [FEATURE_INFO[FEATURES[i]]["label"] for i in order]
    ordered = values[order]
    colours = ["#d64545" if value > 0 else "#1f6feb" for value in ordered]
    span = max(float(np.abs(ordered).max()), 1e-9)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.barh(labels, ordered, color=colours, height=0.65)
    ax.axvline(0, color="#9aa3ad", lw=1)
    for position, value in enumerate(ordered):
        offset = 0.02 * span if value >= 0 else -0.02 * span
        ax.text(value + offset, position, f"{value:+.3f}", va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7, color="#1f2933")
    ax.set_xlim(-span * 1.45, span * 1.45)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("impact on predicted risk", fontsize=8)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner=False)
def load_lime_explainer(path: str):
    """Fit LIME on the full dataset when available, else the background sample.

    LIME needs a training distribution to draw neighbours from; the 100-row
    background sample works but the full 768-row dataset gives a fuller
    neighbourhood. `discretize_continuous=False` is deliberate: LIME's
    discretiser fails on small samples (scipy truncnorm domain error) and plain
    continuous features are easier to read in the chart.
    """
    frame = pd.read_csv(path)
    training = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    training = training.fillna(training.median())
    return LimeTabularExplainer(
        training.to_numpy(dtype=float),
        feature_names=FEATURES,
        class_names=["No diabetes", "Diabetes"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )


def lime_contribution(lime_explainer, row_frame):
    """Local LIME weights for class 1, in FEATURES order, plus LIME's own risk.

    LIME cannot ingest missing values, while the model pipeline imputes them
    internally. Fill the "not measured" sentinels with the same training medians
    the app already reports, so LIME explains the row the model actually scores.
    """
    frame = row_frame[FEATURES].astype(float).copy()
    frame = frame.fillna(pd.Series(MEDIANS))   # DataFrame.fillna(Series) aligns on columns
    values = frame.iloc[0].to_numpy(dtype=float)

    def predict_fn(arr):
        return np.asarray(
            model.predict_proba(pd.DataFrame(arr, columns=FEATURES)), dtype=float)

    explanation = lime_explainer.explain_instance(
        values, predict_fn, labels=(1,), num_features=len(FEATURES),
        num_samples=1000)
    weights = dict(explanation.as_list(label=1))
    series = pd.Series(
        {feature: float(weights.get(feature, 0.0)) for feature in FEATURES},
        dtype=float)
    local_risk = float(explanation.predict_proba[1])
    return series, local_risk


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
        st.success("✅ Model trained and saved.")

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
    with st.spinner("Preparing SHAP explainability…"):
        explainer, explainer_error = safe(
            lambda: shap.Explainer(predict_proba_fn, bg, feature_names=FEATURES),
            SHAP_INIT_TIMEOUT,
            "SHAP explainer setup",
        )

    return model, meta, bg, explainer, explainer_error


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


_GLOBAL_SHAP = {"key": None, "value": None}


def global_explanation(explainer, background: pd.DataFrame):
    """Memoised global SHAP values.

    Deliberately *not* decorated with st.cache_resource: this runs inside a
    worker thread (see run_with_timeout) where Streamlit's cache has no script
    context and would raise. A plain memo is thread-safe enough here because the
    explainer object is stable for the lifetime of the cached model.
    """
    key = (id(explainer), len(background))
    if _GLOBAL_SHAP["key"] != key:
        _GLOBAL_SHAP["key"] = key
        _GLOBAL_SHAP["value"] = explainer(background)
    return _GLOBAL_SHAP["value"]


model, meta, bg, explainer, explainer_error = load()

# Medians used to fill "not measured" values, shown to the user for transparency.
MEDIANS = {
    feature: float(bg[feature].replace(0, np.nan).median())
    for feature in ZERO_AS_MISSING
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

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧑‍⚕️ Patient Information")
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
        st.success("✅ All entered values are in the normal band.")

    st.markdown("---")
    st.markdown("**🩺 Input checks**")
    if issues:
        for level, message in issues:
            {"warning": st.warning, "error": st.error, "info": st.info}[level](message)
    else:
        st.caption("No implausible or missing measurements detected.")

    with st.expander("📋 Reference ranges"):
        for feature in FEATURES:
            st.caption(f"**{feature}**: {FEATURE_INFO[feature]['normal']}")

    st.markdown("---")
    st.caption(f"Model in use: **{meta.get('best_model', 'unknown')}** · "
               f"decision threshold **{THRESHOLD:.2f}**")
    if st.button("🚪 Sign out"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="margin:0">🩺 Diabetes Risk Predictor</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 4px 0">'
    f'<span class="tag">{meta.get("best_model", "unknown")}</span>'
    f'<span class="tag">threshold {THRESHOLD:.2f}</span>'
    '<span class="tag">SHAP + LIME</span></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Estimates the likelihood of diabetes from eight routine measurements. "
    "⚠️ Educational tool only — not a substitute for medical diagnosis.")


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
    st.warning("⚠️ The saved model didn't match these inputs (`"
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
        f"⚠️ This prediction uses **{len(FEATURES) - len(missing_features)} of "
        f"{len(FEATURES)}** measured values — "
        f"{', '.join(missing_features)} were imputed.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Risk Prediction",
    "🧠 SHAP Explainability",
    "📊 Model Performance",
    "🔎 Data Explorer",
    "🆚 Patient Comparison"
])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────
with tab1:
    level = "Low" if risk < .3 else "Moderate" if risk < .6 else "High"
    color = "green" if risk < .3 else "orange" if risk < .6 else "red"
    icon = "✅" if risk < .3 else "⚠️" if risk < .6 else "🚨"

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
            st.metric("Risk Level", level)
            st.markdown(f"### :{color}[{icon} {level} Risk]")
            st.caption(f"Screening threshold: **{THRESHOLD:.2f}**")

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
            st.markdown("#### 🩺 Screening decision")
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
            st.markdown("#### 📉 How certain is this number?")
            if risk_interval:
                low, high, neighbours = risk_interval
                st.markdown(
                    f"The model predicts **{risk:.1%}**, with a 90% interval of "
                    f"**{low:.1%} – {high:.1%}**.")
                fig_ci, ax_ci = plt.subplots(figsize=(6, 1.4))
                ax_ci.errorbar([risk], [0],
                               xerr=[[max(risk - low, 0.0)], [max(high - risk, 0.0)]],
                               fmt="o", color="tab:blue", capsize=6, label="Prediction")
                ax_ci.axvline(THRESHOLD, color="red", ls="--",
                              label=f"Threshold {THRESHOLD:.2f}")
                ax_ci.set_xlim(0, 1)
                ax_ci.set_yticks([])
                ax_ci.set_xlabel("Predicted diabetes risk")
                ax_ci.legend(loc="upper center", ncol=2, fontsize=8)
                st.pyplot(fig_ci)
                plt.close(fig_ci)
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
    st.subheader("🎯 Before / After: improving one risk factor")
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
                    st.warning("⚠️ The comparison could not be scored for this value. "
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

            fig_ba, ax_ba = plt.subplots(figsize=(6.8, 2.2))
            bar_positions = ["Current", "After improvement"]
            bar_values = [risk * 100, next_risk * 100]
            bars = ax_ba.barh(bar_positions, bar_values,
                              color=[BAND_COLOUR[level], "#1f6feb"], height=0.55)
            for bar, value in zip(bars, bar_values):
                inside = value > 14
                ax_ba.text(value - 1.5 if inside else value + 1.5,
                           bar.get_y() + bar.get_height() / 2,
                           f"{value:.1f}%", va="center",
                           ha="right" if inside else "left",
                           color="white" if inside else "#1f2933",
                           fontsize=10, fontweight="bold")
            ax_ba.axvline(THRESHOLD * 100, color="#111827", ls="--", lw=1.5)
            ax_ba.text(THRESHOLD * 100 + 1.0, -0.47,
                       f"threshold {THRESHOLD:.2f}", fontsize=8, color="#111827")
            ax_ba.set_xlim(0, 108)
            ax_ba.set_ylim(1.55, -0.55)
            ax_ba.set_xlabel("Predicted risk (%)", fontsize=9)
            ax_ba.grid(axis="x", color="#eef1f4", lw=0.8)
            ax_ba.set_axisbelow(True)
            ax_ba.tick_params(labelsize=9)
            for side in ("top", "right"):
                ax_ba.spines[side].set_visible(False)
            fig_ba.tight_layout()
            st.pyplot(fig_ba)
            plt.close(fig_ba)

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
    st.subheader("🏆 What to improve first")
    st.markdown(
        "Each modifiable value is moved to its healthy target **one at a time**, then "
        "the scenarios are ranked by the risk that remains. The top row is the single "
        "change with the largest modelled benefit for this patient.")
    rec_rows, combined_risk = improvement_scenarios(cleaned, model)

    with st.container(border=True):
        if not rec_rows:
            st.success(
                "✅ Every modifiable value is already at or below its healthy target, so "
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
        st.markdown("#### 🚩 Clinical flags")
        flags = get_flags(cleaned)
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ No clinical thresholds exceeded. All entered values appear "
                       "within acceptable ranges.")

    with st.container(border=True):
        st.markdown("#### 📋 Patient summary")
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

# ── Tab 2: SHAP Explainability ────────────────────────────────────────────
with tab2:
    st.subheader("🧠 Why did the model make this prediction?")
    st.markdown(
        "This section uses **SHAP** (SHapley Additive exPlanations) to explain "
        "exactly which factors are driving this prediction and by how much. "
        "SHAP values are based on game theory — each feature gets a fair share of "
        "credit or blame for the final prediction."
    )

    if explainer is None:
        st.error(
            "⚠️ **Explanations are temporarily unavailable — your risk score is unaffected.** "
            "The SHAP explainer could not be created, most often because it timed out or "
            "ran out of memory.")
        st.caption(explainer_error or "No further details were captured.")
        with st.expander("Technical details"):
            st.code(explainer_error or "No error message was captured.")
        if st.button("🔁 Try loading SHAP again", key="retry_shap"):
            st.cache_resource.clear()
            st.rerun()
    else:
        sv, shap_error = safe(lambda: explainer(x), SHAP_ROW_TIMEOUT,
                              "SHAP explanation for this patient")

        if sv is None:
            st.warning(
                "⚠️ **This patient's SHAP explanation could not be computed in time.** "
                "The risk score above is still valid. Try reloading the page.")
            with st.expander("Technical details"):
                st.code(shap_error)
        else:
            # ── Waterfall plot ──────────────────────────────────────────────
            st.markdown("#### 📊 SHAP Waterfall Chart")
            st.markdown(
                "This chart starts from the **average prediction** across all patients "
                "and shows how each feature **pushes the risk up (red)** or **pulls it down (blue)** "
                "to arrive at this patient's final predicted risk."
            )
            fig = plt.figure(figsize=(10, 5))
            try:
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig)
            except Exception as exc:  # noqa: BLE001
                st.warning("⚠️ The waterfall chart could not be drawn. "
                           f"({type(exc).__name__}: {exc})")
            finally:
                plt.close(fig)

            st.markdown("---")

            # ── Plain English explanation ───────────────────────────────────
            contrib = pd.DataFrame({
                "Feature": FEATURES,
                "Value":   x.iloc[0].values,
                "Impact":  sv.values[0],
            }).sort_values("Impact", key=abs, ascending=False)

            top_pos = contrib[contrib["Impact"] > 0].head(4)
            top_neg = contrib[contrib["Impact"] < 0].head(4)

            st.markdown("#### 📝 Plain-English Breakdown")
            st.markdown(
                "Below is a detailed explanation of what each factor means for this patient "
                "and why it is increasing or decreasing the predicted risk."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔴 Factors INCREASING Risk")
                if top_pos.empty:
                    st.success("No factors are significantly increasing risk.")
                else:
                    for _, row in top_pos.iterrows():
                        feat   = row["Feature"]
                        val    = row["Value"]
                        impact = row["Impact"]
                        desc   = (explain_missing(feat, MEDIANS.get(feat))
                                  if pd.isna(val) else explain_feature_up(feat, val))
                        with st.expander(f"**{FEATURE_INFO[feat]['label']}** · SHAP: `+{impact:.3f}`", expanded=True):
                            st.markdown(desc)
                            st.caption(f"Normal range: {FEATURE_INFO[feat]['normal']}")

            with col2:
                st.markdown("### 🔵 Factors DECREASING Risk")
                if top_neg.empty:
                    st.info("No factors are significantly decreasing risk.")
                else:
                    for _, row in top_neg.iterrows():
                        feat   = row["Feature"]
                        val    = row["Value"]
                        impact = row["Impact"]
                        desc   = (explain_missing(feat, MEDIANS.get(feat))
                                  if pd.isna(val) else explain_feature_down(feat, val))
                        with st.expander(f"**{FEATURE_INFO[feat]['label']}** · SHAP: `{impact:.3f}`", expanded=True):
                            st.markdown(desc)
                            st.caption(f"Normal range: {FEATURE_INFO[feat]['normal']}")

            st.markdown("---")

            # ── Full impact table ───────────────────────────────────────────
            st.markdown("#### 📋 Complete Feature Impact Table")
            st.markdown(
                "This table ranks all features by how strongly they influenced "
                "this prediction. Positive impact = increases risk. Negative = reduces risk."
            )
            display_df = pd.DataFrame({
                "Feature":      [FEATURE_INFO[f]["label"] for f in contrib["Feature"]],
                "Your Value":   [fmt_value(f, v) for f, v in zip(contrib["Feature"], contrib["Value"])],
                "SHAP Impact":  contrib["Impact"].round(4).values,
                "Direction":    contrib["Impact"].apply(
                    lambda v: "⬆️ Increases risk" if v > 0 else "⬇️ Reduces risk").values,
                "Normal Range": [FEATURE_INFO[f]["normal"] for f in contrib["Feature"]],
            })
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            # ── Stage 4: LIME alongside SHAP ────────────────────────────────
            st.markdown("---")
            st.subheader("🍋 LIME — a second opinion")
            st.markdown(
                "SHAP explains this prediction by fairly distributing credit across the "
                "features. **LIME** works differently: it perturbs this patient's values, "
                "watches how the prediction moves, and fits a simple local model around "
                "them. Because the two methods make different assumptions, comparing them "
                "is a useful honesty check on the explanation.")

            if not LIME_OK:
                st.info("ℹ️ LIME is not installed in this deployment, so only SHAP is shown. "
                        "Add `lime>=0.2.0.1` to `requirements.txt` and reboot the app.")
            else:
                lime_weights = None
                lime_pred = None
                try:
                    lime_explainer = load_lime_explainer(
                        DATA_PATH if os.path.exists(DATA_PATH) else BG_PATH)
                    lime_weights, lime_pred = lime_contribution(lime_explainer, x)
                except Exception as exc:  # noqa: BLE001
                    st.warning("⚠️ LIME could not explain this patient, but the SHAP "
                               f"explanation above is unaffected. ({type(exc).__name__})")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

                if lime_weights is not None:
                    shap_by_feature = dict(zip(contrib["Feature"], contrib["Impact"]))
                    shap_vec = np.array([float(shap_by_feature[f]) for f in FEATURES])
                    lime_vec = lime_weights[FEATURES].to_numpy(dtype=float)

                    rho = rank_correlation(shap_vec, lime_vec)
                    shap_top = FEATURES[int(np.argmax(np.abs(shap_vec)))]
                    lime_top = FEATURES[int(np.argmax(np.abs(lime_vec)))]
                    top_shap = {FEATURES[i] for i in np.argsort(-np.abs(shap_vec))[:3]}
                    top_lime = {FEATURES[i] for i in np.argsort(-np.abs(lime_vec))[:3]}
                    overlap = len(top_shap & top_lime)

                    a1, a2, a3 = st.columns(3)
                    a1.metric(
                        "Importance agreement",
                        "n/a" if rho is None else f"{rho:+.2f}",
                        help="Spearman correlation of the two importance rankings by "
                             "absolute size. +1 means identical order, 0 means unrelated.")
                    a2.metric(
                        "Top-3 overlap", f"{overlap}/3",
                        help="How many features appear in both methods' three strongest.")
                    a3.metric(
                        "Top driver",
                        "agree" if shap_top == lime_top else "differs",
                        help=f"SHAP: {short_label(shap_top)} · "
                             f"LIME: {short_label(lime_top)}")

                    bar_left, bar_right = st.columns(2, gap="large")
                    with bar_left:
                        render_contribution_bars(shap_vec, "SHAP")
                    with bar_right:
                        render_contribution_bars(lime_vec, "LIME")

                    if rho is None:
                        st.info("One of the methods produced no usable ranking for this "
                                "patient, so their agreement cannot be scored.")
                    elif rho >= 0.7:
                        st.success(
                            f"**The two methods largely agree** (importance correlation "
                            f"{rho:+.2f}, {overlap} of 3 top drivers shared). The "
                            "explanation does not depend on SHAP's particular assumptions.")
                    elif rho >= 0.4:
                        st.info(
                            f"**Partial agreement** (importance correlation {rho:+.2f}, "
                            f"{overlap} of 3 top drivers shared). The biggest drivers line "
                            "up, but the methods disagree about some smaller factors.")
                    else:
                        st.warning(
                            f"**The methods weight different features** (importance "
                            f"correlation {rho:+.2f}, {overlap} of 3 top drivers shared). "
                            "LIME fits a local surrogate, so in sparse regions it can latch "
                            "onto a different pattern than the global SHAP decomposition. "
                            "Treat features that both methods rank highly as the robust "
                            "part of the story.")

                    st.caption(
                        f"LIME's local surrogate predicts **{lime_pred:.1%}** for this "
                        f"patient (the full model says {risk:.1%}). Weight scales are not "
                        "directly comparable, and the two methods answer slightly different "
                        "questions: SHAP measures contribution **relative to the dataset "
                        "average**, while LIME measures the local slope **around this "
                        "patient**. That is why the ranking can agree while individual "
                        "signs differ, especially for a patient close to the average.")

                    with st.expander("🔍 Side-by-side importance table"):
                        shap_rank = pd.Series(-np.abs(shap_vec)).rank().astype(int).to_numpy()
                        lime_rank = pd.Series(-np.abs(lime_vec)).rank().astype(int).to_numpy()
                        lime_table = pd.DataFrame({
                            "Feature":   [FEATURE_INFO[f]["label"] for f in FEATURES],
                            "Value":     [fmt_value(f, cleaned[f]) for f in FEATURES],
                            "SHAP":      shap_vec.round(4),
                            "LIME":      lime_vec.round(4),
                            "SHAP rank": shap_rank,
                            "LIME rank": lime_rank,
                            "Direction": [
                                "—" if abs(shap_vec[i]) < 1e-9 or abs(lime_vec[i]) < 1e-9
                                else ("same" if np.sign(shap_vec[i]) == np.sign(lime_vec[i])
                                      else "differs")
                                for i in range(len(FEATURES))],
                        }).sort_values("SHAP rank")
                        st.dataframe(lime_table, hide_index=True,
                                     use_container_width=True)
                        st.caption("Direction compares the sign of the two weights: "
                                   "'same' means both say the value raises (or both say it "
                                   "lowers) risk. A sign difference is expected near the "
                                   "average patient, where SHAP's baseline and LIME's local "
                                   "neighbourhood are almost the same point.")

            st.markdown("---")

            # ── Global importance & beeswarm ───────────────────────────────
            sample = bg.head(100).copy()
            gsv, gsv_error = safe(
                lambda: global_explanation(explainer, sample),
                SHAP_GLOBAL_TIMEOUT, "Global SHAP values")

            if gsv is None:
                st.warning("⚠️ Global feature importance could not be computed, "
                           "but the patient-specific explanation above is unaffected.")
                with st.expander("Technical details"):
                    st.code(gsv_error)
            else:
                st.markdown("#### 🌍 Global Feature Importance")
                st.markdown(
                    f"This chart shows which features matter **most on average across all "
                    f"{len(sample)} sampled patients** in the training data — not just for this "
                    "specific prediction. It gives a general picture of what drives diabetes "
                    "risk in this dataset."
                )
                imp        = pd.Series(np.abs(gsv.values).mean(0), index=FEATURES).sort_values()
                top_global = imp.idxmax()
                fig2, ax   = plt.subplots(figsize=(8, 4))
                imp.plot(kind="barh", ax=ax, color="steelblue")
                ax.set_xlabel("Mean |SHAP value|")
                ax.set_title("Global Feature Importance (mean absolute SHAP)")
                st.pyplot(fig2)
                plt.close(fig2)
                st.info(
                    f"**{FEATURE_INFO[top_global]['label']}** is the most important feature overall — "
                    "it has the biggest average influence on diabetes predictions across "
                    "all patients in the dataset.")

                st.markdown("---")
                st.markdown("#### 🐝 SHAP Beeswarm Plot")
                st.markdown(
                    "Each dot is one patient from the background data. "
                    "The color shows the feature value (red = high, blue = low). "
                    "The position on the x-axis shows whether that value pushed risk up or down. "
                    "This lets you see patterns across the whole dataset at once."
                )
                fig3 = plt.figure(figsize=(10, 5))
                try:
                    shap.plots.beeswarm(gsv, show=False)
                    st.pyplot(fig3)
                except Exception as exc:  # noqa: BLE001
                    st.warning("⚠️ The beeswarm plot could not be drawn. "
                               f"({type(exc).__name__}: {exc})")
                finally:
                    plt.close(fig3)

                # ── Stage 4: SHAP dependence for the top 3 features ─────────
                st.markdown("---")
                st.markdown("#### 📈 SHAP dependence — the three strongest features")
                st.markdown(
                    "Each dot is one background patient: the x-axis is the measured value, "
                    "the y-axis is how much that value pushed that patient's prediction. A "
                    "rising cloud means higher values raise risk, a flat cloud means the "
                    "model barely reacts in that range. The black line is a rolling average "
                    "and the red marker is this patient.")
                imp_series = pd.Series(np.abs(gsv.values).mean(0), index=FEATURES)
                top3 = list(imp_series.sort_values(ascending=False).head(3).index)
                dep_cols = st.columns(3, gap="medium")
                for dep_col, feature in zip(dep_cols, top3):
                    idx = FEATURES.index(feature)
                    xs = sample[feature].astype(float).to_numpy()
                    ys = np.asarray(gsv.values)[:, idx]

                    with dep_col:
                        fig_dep, ax_dep = plt.subplots(figsize=(4.0, 3.0))
                        ax_dep.scatter(xs, ys, s=22, alpha=0.7, color="#1f6feb",
                                       edgecolor="white", linewidth=0.4)
                        order = np.argsort(xs)
                        window = max(4, len(xs) // 8)
                        trend = pd.Series(ys[order]).rolling(
                            window, center=True, min_periods=2).mean()
                        ax_dep.plot(xs[order], trend, color="#111827", lw=1.6)
                        ax_dep.axhline(0, color="#9aa3ad", lw=1)

                        current = cleaned[feature]
                        if pd.notna(current):
                            nearest = int(np.argmin(np.abs(xs - float(current))))
                            ax_dep.scatter([float(current)], [ys[nearest]], s=65,
                                           color="#d64545", edgecolor="white",
                                           linewidth=1.2, zorder=5)
                            ax_dep.axvline(float(current), color="#d64545", ls="--", lw=1)

                        ax_dep.set_title(short_label(feature), fontsize=10,
                                         fontweight="bold")
                        ax_dep.set_xlabel(FEATURE_INFO[feature]["label"], fontsize=7)
                        ax_dep.set_ylabel("SHAP impact", fontsize=7)
                        ax_dep.tick_params(labelsize=7)
                        st.pyplot(fig_dep)
                        plt.close(fig_dep)
                        st.caption(f"Mean |SHAP| **{imp_series[feature]:.3f}** · "
                                   f"{FEATURE_INFO[feature]['normal']}")

                st.caption(
                    "The shape matters more than the exact numbers: a step or a plateau is "
                    "where the model stops treating the value as a graded risk factor. "
                    "Dependence shows association learned from this dataset, not a "
                    "biological dose-response curve.")

                # ── Stage 4: two-feature interaction ────────────────────────
                st.markdown("---")
                st.markdown("#### 🧭 Feature interaction")
                st.markdown(
                    "A dependence plot holds every other value fixed. Here two features "
                    "move together across a grid while the other six stay at this patient's "
                    "values, so you can see whether one changes the effect of the other. "
                    "The dark contour is the screening threshold.")
                i1, i2 = st.columns(2)
                with i1:
                    f_x = st.selectbox(
                        "Horizontal axis", FEATURES,
                        index=FEATURES.index("Glucose"),
                        format_func=lambda f: FEATURE_INFO[f]["label"],
                        key="interact_x")
                with i2:
                    f_y = st.selectbox(
                        "Vertical axis", FEATURES,
                        index=FEATURES.index("BMI"),
                        format_func=lambda f: FEATURE_INFO[f]["label"],
                        key="interact_y")

                if f_x == f_y:
                    st.info("Choose two different features to see an interaction.")
                else:
                    try:
                        grid_axis = {}
                        for feature in (f_x, f_y):
                            series = bg[feature]
                            if feature in ZERO_AS_MISSING:
                                series = series.replace(0, np.nan)
                            observed = series.dropna()
                            if observed.empty:
                                observed = bg[feature]
                            lo, hi = np.nanpercentile(observed, [1, 99])
                            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                                raise ValueError(f"no usable range for {feature}")
                            grid_axis[feature] = np.linspace(float(lo), float(hi), 30)

                        xs = grid_axis[f_x]
                        ys = grid_axis[f_y]
                        xx, yy = np.meshgrid(xs, ys)
                        sweep = pd.DataFrame({
                            feature: np.full(xx.size, float(cleaned[feature]))
                            for feature in FEATURES})
                        sweep[f_x] = xx.ravel()
                        sweep[f_y] = yy.ravel()
                        zz = model.predict_proba(sweep[FEATURES])[:, 1].reshape(xx.shape)

                        fig_int, ax_int = plt.subplots(figsize=(7.2, 4))
                        mesh = ax_int.contourf(xs, ys, zz, levels=20,
                                               cmap="RdYlGn_r", vmin=0, vmax=1)
                        ax_int.contour(xs, ys, zz, levels=[THRESHOLD],
                                       colors="#111827", linewidths=2)
                        ax_int.scatter([float(cleaned[f_x])],
                                       [float(cleaned[f_y])],
                                       s=70, color="#1f6feb", edgecolor="white",
                                       zorder=5, label="This patient")
                        ax_int.plot([], [], color="#111827", lw=2,
                                    label=f"threshold {THRESHOLD:.2f}")
                        fig_int.colorbar(mesh, ax=ax_int, label="Predicted risk")
                        ax_int.set_xlabel(FEATURE_INFO[f_x]["label"], fontsize=8)
                        ax_int.set_ylabel(FEATURE_INFO[f_y]["label"], fontsize=8)
                        ax_int.tick_params(labelsize=8)
                        ax_int.legend(fontsize=8, loc="best")
                        st.pyplot(fig_int)
                        plt.close(fig_int)

                        # Interaction contrast on the four corners of the grid:
                        # does the effect of X differ between low and high Y?
                        effect_x_low_y = (zz[0, -1] - zz[0, 0]) * 100
                        effect_x_high_y = (zz[-1, -1] - zz[-1, 0]) * 100
                        interaction = effect_x_high_y - effect_x_low_y

                        if abs(interaction) < 3:
                            verdict = ("mostly **additive** — moving one of them shifts "
                                       "risk by about the same amount whatever the other "
                                       "is doing.")
                        elif interaction > 0:
                            verdict = (f"**synergistic** here: each extra unit of "
                                       f"{short_label(f_x)} adds {interaction:.1f} more "
                                       f"points of risk when {short_label(f_y)} is high "
                                       "than when it is low.")
                        else:
                            verdict = (f"**offsetting** here: each extra unit of "
                                       f"{short_label(f_x)} adds {abs(interaction):.1f} "
                                       f"fewer points of risk when {short_label(f_y)} is "
                                       "high than when it is low.")

                        st.info(
                            f"At the low end of {short_label(f_y)}, sweeping "
                            f"{short_label(f_x)} across its observed range changes risk by "
                            f"**{effect_x_low_y:+.1f} points**; at the high end it changes by "
                            f"**{effect_x_high_y:+.1f} points**. The gap is "
                            f"**{interaction:+.1f} points**, so the pair is {verdict}")
                        st.caption(
                            "Ranges run from the 1st to the 99th percentile of the "
                            "background sample so a few outliers cannot flatten the colour "
                            "scale. As with every what-if here, the other six values are "
                            "held fixed — an interaction in the model, not proof of one in "
                            "human physiology.")
                    except Exception as exc:  # noqa: BLE001
                        st.warning("⚠️ The interaction map could not be generated. "
                                   f"({type(exc).__name__}: {exc})")

            st.markdown("---")

            # ── What-if analysis ────────────────────────────────────────────
            st.markdown("#### 🔬 What-If Analysis")
            st.markdown(
                "Select a feature below to see how the predicted risk would change "
                "if that value were different — while keeping everything else the same. "
                "This helps identify which values the patient could realistically improve."
            )
            f_feat = st.selectbox("Choose a feature to vary", FEATURES, index=1)

            observed = bg[f_feat].replace(0, np.nan).dropna()
            if observed.empty:
                st.info("Not enough data in the background sample to draw a what-if curve "
                        "for this feature.")
            else:
                try:
                    rng   = np.linspace(float(observed.min()), float(observed.max()), 50)
                    sweep = pd.concat([x] * len(rng), ignore_index=True)
                    sweep[f_feat] = rng
                    risks_sweep = model.predict_proba(sweep)[:, 1]
                    best_val    = rng[np.argmin(risks_sweep)]
                    worst_val   = rng[np.argmax(risks_sweep)]

                    fig4, ax4 = plt.subplots(figsize=(8, 3))
                    ax4.plot(rng, risks_sweep, color="steelblue", linewidth=2)
                    current = cleaned[f_feat]
                    if pd.notna(current):
                        ax4.axvline(current, color="red", ls="--",
                                    label=f"Current: {current:g}")
                    ax4.axvline(best_val, color="green", ls="--",
                                label=f"Lowest risk: {best_val:.1f}")
                    ax4.set_xlabel(FEATURE_INFO[f_feat]["label"])
                    ax4.set_ylabel("Predicted Risk")
                    ax4.set_title(f"How {f_feat} affects predicted diabetes risk")
                    ax4.legend()
                    st.pyplot(fig4)
                    plt.close(fig4)

                    current_txt = (
                        "was not measured" if pd.isna(current)
                        else f"is **{current:g}**")
                    st.info(
                        f"💡 **Insight:** The patient's current "
                        f"**{FEATURE_INFO[f_feat]['label']}** {current_txt}. "
                        f"The lowest predicted risk in the observed range occurs near "
                        f"**{best_val:.1f}**, and the highest near **{worst_val:.1f}**. "
                        f"Normal range for reference: {FEATURE_INFO[f_feat]['normal']}.")
                except Exception as exc:  # noqa: BLE001
                    st.warning("⚠️ The what-if analysis could not be generated. "
                               f"({type(exc).__name__}: {exc})")

# ── Tab 3: Model Performance ──────────────────────────────────────────────
with tab3:
    st.subheader("📊 Model Comparison")
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
        st.warning("⚠️ No model metrics were found in `metrics.json`. Retrain the model to "
                   "regenerate them.")
    else:
        res = pd.DataFrame(results).T.drop(columns="params", errors="ignore")
        st.dataframe(
            res.style.highlight_max(axis=0, color="#cfe8cf").format("{:.4f}"),
            use_container_width=True)

        best_params = (results.get(meta.get("best_model", ""), {}) or {}).get("params")
        if best_params:
            with st.expander("🏆 Best hyper-parameters"):
                st.json(best_params)

        # ── Threshold tuning ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Decision Threshold Tuning")
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
            st.markdown("#### ROC Curve")
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
            st.markdown("#### Confusion Matrix")
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
            "⚠️ **Important note on recall:** In medical screening, missing a diabetic patient "
            "(false negative) is more dangerous than a false alarm (false positive). "
            f"That is why the cut-off was lowered from 0.50 to {THRESHOLD:.2f}: recall rises to "
            f"{AT_THRESHOLD.get('recall', float('nan')):.0%} at the cost of a few extra false "
            "alarms, which is the right trade-off for a screening tool."
        )

# ── Tab 4: Data Explorer ──────────────────────────────────────────────────
with tab4:
    st.subheader("🔎 Dataset Explorer")

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
        st.error("⚠️ " + explorer_error)
        st.caption("The other three tabs are unaffected.")
    else:
        target_col = "Outcome" if "Outcome" in d.columns else "diabetes"

        if target_col not in d.columns:
            st.error("⚠️ The dataset has no `Outcome` (or `diabetes`) column, so the "
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
            st.markdown("#### Correlation Heatmap")
            st.markdown(
                "Shows how strongly each pair of features is related. "
                "Values close to 1 or -1 mean strong correlation. "
                "Glucose and BMI tend to have the strongest correlation with diabetes outcome."
            )
            st.dataframe(d.corr(numeric_only=True).round(2), use_container_width=True)

# ── Tab 5: Patient comparison ─────────────────────────────────────────────
with tab5:
    st.subheader("🆚 Patient Comparison")
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
                    st.caption("⚠️ Patient B has input warnings: "
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
                st.caption("SHAP is unavailable, so the drivers cannot be listed. "
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
                            order = np.argsort(-np.abs(vector))[:3]
                            for idx in order:
                                direction = "raises" if vector[idx] > 0 else "lowers"
                                st.markdown(
                                    f"- {short_label(FEATURES[idx])}: "
                                    f"`{vector[idx]:+.3f}` ({direction} risk)")
                        except Exception as exc:  # noqa: BLE001
                            st.caption(f"Drivers unavailable ({type(exc).__name__}).")

            st.caption(
                "As everywhere else in this app, the scores are model estimates from eight "
                "routine measurements, not diagnoses. A difference between two profiles "
                "shows how the model responds to their inputs, not a treatment effect.")

st.divider()
st.caption(
    "Trained on the Pima Indians Diabetes Dataset (adult women of Pima heritage, aged 21+). "
    "Class imbalance handled with balanced class weights (no SMOTE). "
    "SHAP used for explainability; the decision threshold is tuned for high recall. "
    "This tool is for educational purposes only and is not a medical diagnosis."
)
