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
* The risk figure is a Plotly gauge with the three risk bands and the tuned
  threshold marked; a plain progress bar is used only if Plotly is unavailable.
* Every input carries a colour-coded status pill (normal / borderline /
  abnormal / not measured) derived from clinical reference bands, so an
  out-of-range value is visible at a glance rather than buried in a warning list.
* A "before / after" panel re-scores the patient with one modifiable value moved
  to a healthy target and shows the resulting drop in risk.
* Presentation is grouped into cards (`st.container(border=True)`) with a single
  injected stylesheet, so the page reads as one designed surface.
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
import numpy as np
import pandas as pd
import shap
import streamlit as st

try:  # optional: only needed if the saved model is an XGBoost pipeline
    import xgboost
    XGBOOST_VERSION = xgboost.__version__
except Exception:  # noqa: BLE001
    XGBOOST_VERSION = None

try:  # Stage 3: gauge + comparison charts. The app degrades to native widgets.
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:  # noqa: BLE001
    go = None
    PLOTLY_OK = False

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
    """Plotly gauge: three risk bands plus a marker at the tuned threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        number={"suffix": "%", "font": {"size": 32, "color": "#1f2933"},
                "valueformat": ".1f"},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9aa3ad",
                     "showticklabels": False},
            "bar": {"color": BAND_COLOUR[level], "thickness": 0.26},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#e6f4ea"},
                {"range": [30, 60], "color": "#fff4d6"},
                {"range": [60, 100], "color": "#fde8e6"},
            ],
            "threshold": {"line": {"color": "#111827", "width": 3},
                          "thickness": 0.78, "value": threshold * 100},
        },
    ))
    fig.update_layout(
        height=250,
        margin={"l": 38, "r": 38, "t": 14, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans serif", "color": "#1f2933"},
    )
    return fig



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
    '<span class="tag">SHAP explainability</span></div>',
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

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Risk Prediction",
    "🧠 SHAP Explainability",
    "📊 Model Performance",
    "🔎 Data Explorer"
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
            if PLOTLY_OK:
                st.plotly_chart(risk_gauge(risk, THRESHOLD, level),
                                use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.metric("Estimated Diabetes Risk", f"{risk:.1%}")
                st.progress(risk)
            if risk_interval:
                st.caption(f"90% interval: **{risk_interval[0]:.1%} – {risk_interval[1]:.1%}** "
                           f"· {risk_interval[2]} similar patients")
            st.caption(
                "Bands: 0–30% low · 30–60% moderate · 60–100% high. "
                + (f"The dark line marks the screening threshold ({THRESHOLD:.2f})."
                   if PLOTLY_OK else
                   f"Screening threshold: {THRESHOLD:.2f}."))

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

    MODIFIABLE = {
        "Glucose":        (100, "mg/dL",    "fasting glucose around 100 mg/dL"),
        "BMI":            (23.0, "kg/m²",   "BMI inside the healthy 18.5–24.9 band"),
        "BloodPressure":  (75, "mm Hg",     "diastolic pressure around 75 mm Hg"),
        "Insulin":        (80, "mu U/ml",   "insulin in the middle of the normal range"),
        "SkinThickness":  (20, "mm",        "skin fold around 20 mm"),
    }

    with st.container(border=True):
        next_risk = None
        ctl, info = st.columns([1, 2], gap="large")

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
                      delta=f"{delta * 100:+.1f} pts")
            e3.metric("Relative change",
                      f"{delta / risk:.0%}" if risk > 1e-6 else "n/a")

            if PLOTLY_OK:
                fig_ba = go.Figure(go.Bar(
                    x=[risk * 100, next_risk * 100],
                    y=["Current", "After improvement"],
                    orientation="h",
                    marker_color=[BAND_COLOUR[level], "#1f6feb"],
                    text=[f"{risk:.1%}", f"{next_risk:.1%}"],
                    textposition="inside",
                    insidetextanchor="end",
                    textfont={"color": "#ffffff", "size": 13},
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                ))
                fig_ba.add_vline(
                    x=THRESHOLD * 100, line_dash="dash", line_color="#111827",
                    annotation_text=f"threshold {THRESHOLD:.2f}",
                    annotation_position="top")
                fig_ba.update_layout(
                    height=230,
                    margin={"l": 10, "r": 40, "t": 34, "b": 26},
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis={"range": [0, 108], "title": "Predicted risk (%)",
                           "showgrid": True, "gridcolor": "#eef1f4"},
                    yaxis={"autorange": "reversed"},
                    showlegend=False,
                )
                st.plotly_chart(fig_ba, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.progress(risk)
                st.caption(f"After improvement: **{next_risk:.1%}**")

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

st.divider()
st.caption(
    "Trained on the Pima Indians Diabetes Dataset (adult women of Pima heritage, aged 21+). "
    "Class imbalance handled with balanced class weights (no SMOTE). "
    "SHAP used for explainability; the decision threshold is tuned for high recall. "
    "This tool is for educational purposes only and is not a medical diagnosis."
)
