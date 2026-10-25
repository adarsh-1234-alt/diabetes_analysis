"""Train, compare and save the diabetes model.

Four candidate models are compared with 5-fold stratified cross-validation:
Logistic Regression, Random Forest, Gradient Boosting and XGBoost.

Imbalance is handled with class weights (`class_weight="balanced"`, or
`scale_pos_weight` for XGBoost) — **no SMOTE**. Synthetic oversampling needs the
training distribution at inference time, which is what broke deployment before;
class weights are baked into the fitted model instead and need no extra
dependency.

After the best model is chosen, the decision threshold is tuned on out-of-fold
predictions to favour recall (healthcare: a missed diabetic is worse than a false
alarm), and the resulting operating point is saved alongside the usual metrics.

Artefacts are written atomically, so a crash can never leave a half-written
model behind.

Usage: python train_model.py
"""
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - deployment safety net
    xgboost, XGBClassifier = None, None

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "diabetes.csv")
MODEL_PATH   = os.path.join(BASE_DIR, "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
BG_PATH      = os.path.join(BASE_DIR, "background.csv")

FEATURES        = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
TARGET          = "Outcome"
RANDOM_STATE    = 42
N_JOBS          = -1

# Threshold tuning: sweep this range and prefer the highest recall whose
# precision is still acceptable for a screening tool.
THRESHOLD_GRID  = np.round(np.arange(0.05, 0.951, 0.01), 2)
MIN_PRECISION   = 0.50
SWEEP_STEP      = 0.05


def fail(message, code=1):
    print(f"\n[ERROR] {message}", flush=True)
    sys.exit(code)


def atomic_write(path, writer):
    """Write via a temp file + os.replace so readers never see a partial file."""
    tmp = f"{path}.tmp"
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Load & validate ─────────────────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(DATA_PATH):
        fail(f"Could not find the dataset at:\n   {DATA_PATH}\n"
             "   Put 'diabetes.csv' next to this script and run again.")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as exc:  # noqa: BLE001
        fail(f"Could not read {DATA_PATH} - {type(exc).__name__}: {exc}")

    if df.empty:
        fail("diabetes.csv is empty.")

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        fail(f"diabetes.csv is missing required columns: {missing}\n"
             f"   Found columns: {list(df.columns)}")

    df = df.copy()
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    if df[TARGET].isna().any():
        fail("The Outcome column contains non-numeric / missing values.")

    # 0 is not a physically possible measurement for these five columns.
    sentinel = int((df[ZERO_AS_MISSING] == 0).sum().sum())
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    dropped = int(df[FEATURES].isna().all(axis=1).sum())
    if dropped:
        print(f"[WARN] Dropping {dropped} row(s) with no usable measurements.", flush=True)
        df = df.dropna(subset=FEATURES, how="all")

    if len(df) < 50:
        fail(f"Only {len(df)} usable rows - too few to train a reliable model.")
    if df[TARGET].nunique() < 2:
        fail("The Outcome column has only one class; cannot train a classifier.")

    print(f"[OK] Loaded {len(df)} rows from {os.path.basename(DATA_PATH)} "
          f"({sentinel} sentinel zeros treated as missing).", flush=True)
    print(f"   Missing per column: "
          f"{df[ZERO_AS_MISSING].isna().sum().to_dict()}", flush=True)
    return df


df = load_dataset()
X, y = df[FEATURES], df[TARGET].astype(int)

print(f"Class distribution (no resampling): {y.value_counts().to_dict()}", flush=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


# ── Pipeline builder ────────────────────────────────────────────────────────
# Plain sklearn Pipeline: every candidate carries its own class weighting, so
# there is no sampler step (and no imblearn dependency) in the saved model.
def pipe(clf):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("clf", clf),
    ])


# ── Model candidates ────────────────────────────────────────────────────────
candidates = {
    "Logistic Regression": (
        pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
        {"clf__C": [0.01, 0.1, 1, 10]}
    ),
    "Random Forest": (
        pipe(RandomForestClassifier(random_state=RANDOM_STATE,
                                    class_weight="balanced", n_jobs=1)),
        {
            "clf__n_estimators":     [300],
            "clf__max_depth":        [4, 6, None],
            "clf__min_samples_leaf": [2, 5],
        }
    ),
    # sklearn's gradient boosting has no class_weight, so the histogram variant
    # is used: same family, balanced weights supported, far lighter on a
    # Streamlit Cloud free-tier container.
    "Gradient Boosting": (
        pipe(HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, class_weight="balanced",
            early_stopping=False, max_iter=300)),
        {
            "clf__max_iter":      [200, 400],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth":     [3, 5],
        }
    ),
}

# XGBoost weights the positive class with scale_pos_weight (= negatives /
# positives), its equivalent of class_weight="balanced".
if XGBClassifier is not None:
    scale_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    candidates["XGBoost"] = (
        pipe(XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,                 # GridSearchCV already parallelises
            tree_method="hist",       # low memory, fast on small data
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )),
        {
            "clf__n_estimators":  [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth":     [3, 5],
        }
    )
    print(f"[OK] XGBoost {xgboost.__version__} available "
          f"(scale_pos_weight={scale_pos_weight:.3f}).", flush=True)
else:
    print("[WARN] xgboost is not installed - training the other three models. "
          "Run `pip install xgboost` to include it.", flush=True)


# ── Threshold tuning ────────────────────────────────────────────────────────
def threshold_rows(y_true, probs):
    """Precision/recall/F1/F2 for every threshold in THRESHOLD_GRID."""
    rows = []
    for thresh in THRESHOLD_GRID:
        pred = (probs >= thresh).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        beta2 = 4.0
        f2 = ((1 + beta2) * precision * recall / (beta2 * precision + recall)) if (precision + recall) else 0.0

        rows.append({"threshold": float(thresh), "precision": precision,
                     "recall": recall, "f1": f1, "f2": f2})
    return rows


def choose_threshold(rows, min_precision=MIN_PRECISION):
    """Highest recall whose precision clears the floor; F2 if nothing does."""
    eligible = [r for r in rows if r["precision"] >= min_precision and r["recall"] > 0]
    if eligible:
        best = max(eligible, key=lambda r: (r["recall"], r["precision"], r["threshold"]))
        return best, f"max recall with precision >= {min_precision:.2f}"

    best = max(rows, key=lambda r: (r["f2"], r["recall"]))
    return best, "max F2 score (no threshold cleared the precision floor)"


def scores_at(y_true, probs, thresh):
    pred = (probs >= thresh).astype(int)
    return {
        "accuracy":  round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, pred, zero_division=0)), 4),
    }


# ── Train & evaluate ────────────────────────────────────────────────────────
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
results, best_name, best_est, best_auc = {}, None, None, -1.0

for name, (pipeline, grid) in candidates.items():
    print(f"\nTraining {name}...", flush=True)
    try:
        gs = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc",
                          n_jobs=N_JOBS).fit(X_tr, y_tr)
        prob = gs.predict_proba(X_te)[:, 1]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {name} failed and was skipped - "
              f"{type(exc).__name__}: {exc}", flush=True)
        continue

    pred = (prob >= 0.5).astype(int)
    results[name] = {
        "cv_auc":    round(float(gs.best_score_), 4),
        "test_auc":  round(float(roc_auc_score(y_te, prob)), 4),
        "accuracy":  round(float(accuracy_score(y_te, pred)), 4),
        "precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_te, pred, zero_division=0)), 4),
        "params":    {k: str(v) for k, v in gs.best_params_.items()},
    }
    print(f"   CV AUC {results[name]['cv_auc']:.4f} | "
          f"test AUC {results[name]['test_auc']:.4f} | "
          f"recall@0.5 {results[name]['recall']:.4f}", flush=True)

    if gs.best_score_ > best_auc:
        best_name, best_est, best_auc = name, gs.best_estimator_, gs.best_score_

if best_est is None:
    fail("Every candidate model failed to train. Check the messages above.")

# ── Tune the decision threshold on out-of-fold predictions ──────────────────
# Out-of-fold probabilities come from refitting on 4/5 of the training data, so
# the threshold is not chosen on the test set (that would be optimistic).
print(f"\nTuning the decision threshold for {best_name}...", flush=True)
try:
    oof = cross_val_predict(best_est, X_tr, y_tr, cv=cv,
                            method="predict_proba", n_jobs=N_JOBS)[:, 1]
    sweep = threshold_rows(y_tr.to_numpy(), oof)
    chosen, rule = choose_threshold(sweep)
    threshold = float(chosen["threshold"])
    print(f"[OK] Threshold {threshold:.2f} ({rule}) - "
          f"OOF recall {chosen['recall']:.3f}, precision {chosen['precision']:.3f}",
          flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[WARN] Threshold tuning failed ({type(exc).__name__}: {exc}); "
          "falling back to 0.50.", flush=True)
    threshold, rule, sweep = 0.5, "fallback to the default 0.50", []

# ── Save artefacts ──────────────────────────────────────────────────────────
prob        = best_est.predict_proba(X_te)[:, 1]
fpr, tpr, _ = roc_curve(y_te, prob)
meta = {
    "best_model": best_name,
    "results":    results,
    "threshold":  threshold,
    "threshold_rule": rule,
    "at_threshold": scores_at(y_te, prob, threshold),
    "at_default":   scores_at(y_te, prob, 0.5),
    "threshold_sweep": [
        {k: round(float(v), 4) for k, v in row.items()}
        for row in sweep
        if abs(row["threshold"] / SWEEP_STEP - round(row["threshold"] / SWEEP_STEP)) < 1e-9
    ],
    "confusion":  confusion_matrix(y_te, (prob >= threshold).astype(int)).tolist(),
    "roc":        {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    "n_train":    int(len(X_tr)),
    "n_test":     int(len(X_te)),
    "features":   FEATURES,
    "imbalance":  "class_weight='balanced' (scale_pos_weight for XGBoost), no SMOTE",
    # Used by app.py to detect artefacts from an incompatible environment.
    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__ if xgboost is not None else None,
    "trained_at":  datetime.now().isoformat(timespec="seconds"),
}

# Background sample for SHAP — raw (un-imputed) rows, so the explainer sees the
# same value distribution the model was trained on.
sample_n = min(100, len(X_tr))
background = X_tr.sample(sample_n, random_state=1)

atomic_write(BG_PATH, lambda p: background.to_csv(p, index=False))
atomic_write(METRICS_PATH,
             lambda p: json.dump(meta, open(p, "w", encoding="utf-8"), indent=2))
# Model is written last: if anything above fails, the old model stays loadable.
atomic_write(MODEL_PATH, lambda p: joblib.dump(best_est, p))

print(f"\n[OK] Saved best model: {best_name}", flush=True)
print(f"   threshold  -> {threshold:.2f} ({rule})", flush=True)
print(f"   recall     -> {meta['at_threshold']['recall']:.4f} "
      f"(vs {meta['at_default']['recall']:.4f} at 0.50)", flush=True)
print(f"   precision  -> {meta['at_threshold']['precision']:.4f} "
      f"(vs {meta['at_default']['precision']:.4f} at 0.50)", flush=True)
print(f"   model      -> {MODEL_PATH}", flush=True)
print(f"   metrics    -> {METRICS_PATH}", flush=True)
print(f"   background -> {BG_PATH} ({len(background)} rows)", flush=True)
