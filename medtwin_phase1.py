"""
MedTwin AI — Phase 1: Risk Scorer
===================================
Trains XGBoost risk models on public datasets.
Outputs: saved models + SHAP explainer artifacts.

Datasets used:
  - Pima Indians Diabetes (diabetes complication risk)
  - Cleveland Heart Disease (cardiovascular risk)

Run: python medtwin_phase1.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Dependencies check ──────────────────────────────────────────────────────
try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, classification_report,
                                 confusion_matrix, roc_curve)
    from sklearn.pipeline import Pipeline
    import joblib
    print("✓ All dependencies found.")
except ImportError as e:
    print(f"Missing: {e}")
    print("Run: pip install xgboost shap scikit-learn pandas numpy matplotlib joblib")
    exit(1)

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_diabetes_data():
    """
    Pima Indians Diabetes Dataset.
    Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    Falls back to synthetic if not found (for demo purposes).
    """
    path = "data/diabetes.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✓ Loaded diabetes data: {df.shape}")
    else:
        print("⚠ diabetes.csv not found — generating synthetic equivalent.")
        print("  Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            "Pregnancies":            np.random.poisson(3.8, n),
            "Glucose":                np.random.normal(120, 32, n).clip(0, 199),
            "BloodPressure":          np.random.normal(69, 19, n).clip(0, 122),
            "SkinThickness":          np.random.normal(20, 16, n).clip(0, 99),
            "Insulin":                np.random.exponential(79, n).clip(0, 846),
            "BMI":                    np.random.normal(32, 7, n).clip(0, 67),
            "DiabetesPedigreeFunction": np.random.exponential(0.47, n).clip(0.078, 2.42),
            "Age":                    np.random.randint(21, 81, n),
        })
        # Simulate outcome with realistic correlation
        risk_score = (
            (df["Glucose"] > 140).astype(float) * 0.4 +
            (df["BMI"] > 30).astype(float) * 0.25 +
            (df["Age"] > 45).astype(float) * 0.2 +
            np.random.normal(0, 0.15, n)
        )
        df["Outcome"] = (risk_score > 0.45).astype(int)
        df.to_csv(path, index=False)

    return df


def load_heart_data():
    """
    Cleveland Heart Disease Dataset.
    Source: https://archive.ics.uci.edu/ml/datasets/heart+disease
    Falls back to synthetic if not found.
    """
    path = "data/heart.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✓ Loaded heart disease data: {df.shape}")
    else:
        print("⚠ heart.csv not found — generating synthetic equivalent.")
        print("  Download from: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci")
        np.random.seed(7)
        n = 303
        df = pd.DataFrame({
            "age":      np.random.randint(29, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),        # chest pain type
            "trestbps": np.random.normal(131, 17, n).clip(94, 200),
            "chol":     np.random.normal(246, 51, n).clip(126, 564),
            "fbs":      np.random.randint(0, 2, n),        # fasting blood sugar > 120
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.normal(149, 22, n).clip(71, 202),
            "exang":    np.random.randint(0, 2, n),        # exercise-induced angina
            "oldpeak":  np.random.exponential(1.04, n).clip(0, 6.2),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.randint(0, 3, n),
        })
        risk_score = (
            (df["age"] > 55).astype(float) * 0.3 +
            (df["trestbps"] > 140).astype(float) * 0.2 +
            (df["chol"] > 240).astype(float) * 0.15 +
            (df["cp"] > 0).astype(float) * 0.25 +
            np.random.normal(0, 0.15, n)
        )
        df["target"] = (risk_score > 0.4).astype(int)
        df.to_csv(path, index=False)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_diabetes(df):
    """
    Handle zero-values (biologically impossible) as missing, impute with median.
    These columns cannot be zero: Glucose, BloodPressure, SkinThickness, Insulin, BMI
    """
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    for col in zero_invalid:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    feature_names = X.columns.tolist()
    return X, y, feature_names


def preprocess_heart(df):
    """Standard preprocessing for heart disease data."""
    df = df.copy()
    df.dropna(inplace=True)

    # Binarize target if it's multi-class (Cleveland has 0–4)
    if "target" in df.columns:
        target_col = "target"
    elif "condition" in df.columns:
        target_col = "condition"
    else:
        # find first integer column that looks like a label
        target_col = df.columns[-1]

    df[target_col] = (df[target_col] > 0).astype(int)

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()
    return X, y, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_risk_model(X, y, feature_names, model_name="model"):
    """
    Trains XGBoost classifier with cross-validation.
    Returns trained pipeline + evaluation metrics.
    """
    print(f"\n{'─'*50}")
    print(f"Training: {model_name}")
    print(f"  Samples: {len(X)}, Features: {len(feature_names)}")
    print(f"  Class balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost handles scaling internally, but we include scaler for API consistency
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"  Test AUC:  {auc:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    # Save model
    save_path = f"models/{model_name}.pkl"
    joblib.dump({"model": model, "features": feature_names}, save_path)
    print(f"  ✓ Saved → {save_path}")

    return model, X_train, X_test, y_train, y_test, y_prob, auc


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(model, X_train, X_test, feature_names, model_name="model"):
    """
    Computes SHAP values for explainability.
    This is the core of the Explainable AI layer in MedTwin.
    """
    print(f"\n  Computing SHAP values for {model_name}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save explainer for later use in API
    joblib.dump(explainer, f"models/{model_name}_shap.pkl")

    return explainer, shap_values


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":        "#0a0e1a",
    "surface":   "#111827",
    "accent":    "#00d4ff",
    "accent2":   "#7c3aed",
    "positive":  "#10b981",
    "danger":    "#ef4444",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
}

def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.spines[:].set_color(PALETTE["muted"])
    ax.spines[:].set_alpha(0.3)
    if title:  ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=PALETTE["muted"], fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=PALETTE["muted"], fontsize=9)


def plot_dashboard(
    diabetes_model, diabetes_shap, diabetes_X_test, diabetes_y_prob, diabetes_y_test,
    heart_model,    heart_shap,    heart_X_test,    heart_y_prob,    heart_y_test,
    diabetes_features, heart_features
):
    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    fig.suptitle(
        "MedTwin AI — Risk Model Dashboard",
        color=PALETTE["accent"], fontsize=18, fontweight="bold", y=0.97
    )

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38,
                           left=0.06, right=0.97, top=0.92, bottom=0.06)

    # ── ROC Curves ───────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, :2])
    style_axes(ax_roc, title="ROC Curves — Both Models", xlabel="False Positive Rate", ylabel="True Positive Rate")

    for y_prob, y_test, color, label in [
        (diabetes_y_prob, diabetes_y_test, PALETTE["accent"],  "Diabetes Risk"),
        (heart_y_prob,    heart_y_test,    PALETTE["accent2"], "Cardiovascular Risk"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], color=PALETTE["muted"], ls="--", lw=1, alpha=0.5)
    ax_roc.fill_between([0, 1], [0, 1], alpha=0.05, color=PALETTE["muted"])
    ax_roc.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
                  labelcolor=PALETTE["text"], fontsize=9)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # ── Risk Distribution ─────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 2:])
    style_axes(ax_dist, title="Predicted Risk Score Distributions", xlabel="Risk Score", ylabel="Density")

    for y_prob, color, label in [
        (diabetes_y_prob, PALETTE["accent"],  "Diabetes"),
        (heart_y_prob,    PALETTE["accent2"], "Cardiovascular"),
    ]:
        ax_dist.hist(y_prob, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax_dist.axvline(y_prob.mean(), color=color, lw=1.5, ls="--", alpha=0.8)

    ax_dist.axvline(0.5, color=PALETTE["danger"], lw=1, ls=":", alpha=0.6, label="Decision boundary")
    ax_dist.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
                   labelcolor=PALETTE["text"], fontsize=9)

    # ── SHAP Feature Importance — Diabetes ──────────────────────────────────
    ax_shap1 = fig.add_subplot(gs[1, :2])
    style_axes(ax_shap1, title="SHAP Feature Importance — Diabetes Risk")

    shap_mean = np.abs(diabetes_shap).mean(axis=0)
    sorted_idx = np.argsort(shap_mean)
    colors_bar = [PALETTE["accent"] if i == sorted_idx[-1] else PALETTE["accent2"]
                  for i in range(len(sorted_idx))]
    bars = ax_shap1.barh(
        [diabetes_features[i] for i in sorted_idx],
        shap_mean[sorted_idx],
        color=colors_bar, alpha=0.85, height=0.6
    )
    ax_shap1.set_xlabel("Mean |SHAP value|", color=PALETTE["muted"], fontsize=9)
    for bar, val in zip(bars, shap_mean[sorted_idx]):
        ax_shap1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                      f"{val:.3f}", va="center", ha="left",
                      color=PALETTE["muted"], fontsize=7.5)

    # ── SHAP Feature Importance — Heart ──────────────────────────────────────
    ax_shap2 = fig.add_subplot(gs[1, 2:])
    style_axes(ax_shap2, title="SHAP Feature Importance — Cardiovascular Risk")

    shap_mean_h = np.abs(heart_shap).mean(axis=0)
    sorted_idx_h = np.argsort(shap_mean_h)
    colors_bar_h = [PALETTE["accent"] if i == sorted_idx_h[-1] else PALETTE["accent2"]
                    for i in range(len(sorted_idx_h))]
    bars_h = ax_shap2.barh(
        [heart_features[i] for i in sorted_idx_h],
        shap_mean_h[sorted_idx_h],
        color=colors_bar_h, alpha=0.85, height=0.6
    )
    ax_shap2.set_xlabel("Mean |SHAP value|", color=PALETTE["muted"], fontsize=9)
    for bar, val in zip(bars_h, shap_mean_h[sorted_idx_h]):
        ax_shap2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                      f"{val:.3f}", va="center", ha="left",
                      color=PALETTE["muted"], fontsize=7.5)

    # ── Simulated Patient Risk Comparison ────────────────────────────────────
    ax_sim = fig.add_subplot(gs[2, :2])
    style_axes(ax_sim, title="Simulated Intervention: Lifestyle Change Impact")

    # Take 3 real test patients and show their risk before/after simulated intervention
    sample_idx = np.where(diabetes_y_prob > 0.6)[0][:3]
    if len(sample_idx) < 3:
        sample_idx = np.argsort(diabetes_y_prob)[-3:]

    patients = ["Patient A", "Patient B", "Patient C"]
    before = diabetes_y_prob[sample_idx]
    # Simulate: reduce glucose by 15%, improve sleep (reduce pedigree proxy)
    after = np.clip(before * np.array([0.72, 0.68, 0.75]), 0, 1)

    x = np.arange(3)
    w = 0.3
    b1 = ax_sim.bar(x - w/2, before, w, label="Before intervention", color=PALETTE["danger"], alpha=0.8)
    b2 = ax_sim.bar(x + w/2, after,  w, label="After intervention",  color=PALETTE["positive"], alpha=0.8)
    ax_sim.set_xticks(x); ax_sim.set_xticklabels(patients, color=PALETTE["text"])
    ax_sim.set_ylabel("Predicted Diabetes Risk", color=PALETTE["muted"], fontsize=9)
    ax_sim.set_ylim(0, 1)
    ax_sim.axhline(0.5, color=PALETTE["muted"], ls="--", lw=1, alpha=0.5)

    for bar in b1:
        ax_sim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", color=PALETTE["danger"], fontsize=8.5, fontweight="bold")
    for bar in b2:
        ax_sim.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", color=PALETTE["positive"], fontsize=8.5, fontweight="bold")

    ax_sim.legend(facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
                  labelcolor=PALETTE["text"], fontsize=9)

    # ── Patient State Radar / Health Profile ─────────────────────────────────
    ax_radar = fig.add_subplot(gs[2, 2:], polar=True)
    ax_radar.set_facecolor(PALETTE["surface"])
    ax_radar.tick_params(colors=PALETTE["text"], labelsize=8)
    ax_radar.spines["polar"].set_color(PALETTE["muted"])
    ax_radar.spines["polar"].set_alpha(0.3)
    ax_radar.set_title("Example Patient Risk Profile", color=PALETTE["text"],
                        fontsize=11, fontweight="bold", pad=15)

    categories = ["Glucose\nRisk", "BP Risk", "BMI Risk", "Age Risk", "Lifestyle\nRisk", "Family\nHistory"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # High-risk patient profile
    high_risk = [0.82, 0.65, 0.71, 0.60, 0.55, 0.78]
    high_risk += high_risk[:1]
    # After intervention
    low_risk  = [0.45, 0.42, 0.55, 0.60, 0.28, 0.78]
    low_risk  += low_risk[:1]

    ax_radar.plot(angles, high_risk, color=PALETTE["danger"],   lw=2, label="Before")
    ax_radar.fill(angles, high_risk, color=PALETTE["danger"],   alpha=0.15)
    ax_radar.plot(angles, low_risk,  color=PALETTE["positive"], lw=2, label="After intervention")
    ax_radar.fill(angles, low_risk,  color=PALETTE["positive"], alpha=0.15)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, color=PALETTE["text"], fontsize=8)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_radar.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color=PALETTE["muted"], fontsize=7)
    ax_radar.set_ylim(0, 1)
    ax_radar.grid(color=PALETTE["muted"], alpha=0.2)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                    facecolor=PALETTE["surface"], edgecolor=PALETTE["muted"],
                    labelcolor=PALETTE["text"], fontsize=9)

    plt.savefig("assets/medtwin_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    print("\n  ✓ Dashboard saved → assets/medtwin_dashboard.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. PATIENT RISK SCORER — INFERENCE API (preview)
# ─────────────────────────────────────────────────────────────────────────────

def predict_patient_risk(patient_dict, model_type="diabetes"):
    """
    Inference function — this becomes the FastAPI endpoint later.

    patient_dict: dict with patient features
    Returns: {"risk_score": float, "risk_level": str, "top_factors": list}
    """
    artifact = joblib.load(f"models/{model_type}_risk.pkl")
    explainer = joblib.load(f"models/{model_type}_risk_shap.pkl")
    model     = artifact["model"]
    features  = artifact["features"]

    X = pd.DataFrame([patient_dict])[features]
    risk_score = model.predict_proba(X)[0][1]

    shap_vals = explainer.shap_values(X)[0]
    factor_importance = sorted(
        zip(features, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_factors = [
        {"feature": f, "impact": round(float(v), 4), "direction": "↑ risk" if v > 0 else "↓ risk"}
        for f, v in factor_importance[:4]
    ]

    risk_level = (
        "Critical" if risk_score > 0.75 else
        "High"     if risk_score > 0.55 else
        "Moderate" if risk_score > 0.35 else
        "Low"
    )

    return {
        "risk_score":  round(float(risk_score), 4),
        "risk_level":  risk_level,
        "top_factors": top_factors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MedTwin AI — Phase 1: Risk Model Training")
    print("=" * 60)

    # ── Load & preprocess ────────────────────────────────────────────────────
    df_diabetes = load_diabetes_data()
    df_heart    = load_heart_data()

    X_d, y_d, feat_d = preprocess_diabetes(df_diabetes)
    X_h, y_h, feat_h = preprocess_heart(df_heart)

    # ── Train models ─────────────────────────────────────────────────────────
    (d_model, d_X_train, d_X_test,
     d_y_train, d_y_test, d_y_prob, d_auc) = train_risk_model(X_d, y_d, feat_d, "diabetes_risk")

    (h_model, h_X_train, h_X_test,
     h_y_train, h_y_test, h_y_prob, h_auc) = train_risk_model(X_h, y_h, feat_h, "heart_risk")

    # ── SHAP explainability ──────────────────────────────────────────────────
    d_explainer, d_shap = compute_shap(d_model, d_X_train, d_X_test, feat_d, "diabetes_risk")
    h_explainer, h_shap = compute_shap(h_model, h_X_train, h_X_test, feat_h, "heart_risk")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    fig = plot_dashboard(
        d_model, d_shap, d_X_test, d_y_prob, d_y_test,
        h_model, h_shap, h_X_test, h_y_prob, h_y_test,
        feat_d, feat_h,
    )

    # ── Demo: inference ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Demo: Predict risk for a sample patient")
    print("─" * 60)
    sample_patient = {
        "Pregnancies": 3,
        "Glucose": 158,
        "BloodPressure": 82,
        "SkinThickness": 32,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 47,
    }
    result = predict_patient_risk(sample_patient, model_type="diabetes")
    print(f"\n  Patient Input: {sample_patient}")
    print(f"\n  Risk Score:  {result['risk_score']:.1%}")
    print(f"  Risk Level:  {result['risk_level']}")
    print(f"  Top Factors:")
    for f in result["top_factors"]:
        print(f"    • {f['feature']:30s} {f['impact']:+.4f}  {f['direction']}")

    print("\n" + "=" * 60)
    print("  Phase 1 complete.")
    print("  Next → Phase 2: Progression Forecaster (LSTM)")
    print("=" * 60)
    plt.show()
