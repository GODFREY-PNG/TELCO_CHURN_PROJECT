"""
main.py
FastAPI application exposing the churn prediction model.

Two ways to use it:
  - Single customer: POST /predict with one customer's data
  - Batch scoring:   POST /predict/batch with a CSV upload
  - Charts:          GET  /charts/{chart_name} for frontend visualizations

Run with:
    uvicorn main:app --reload
Then open: http://localhost:8000
"""

import os
import sys
import pickle
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Scripts folder sits next to main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
import config
from feature_engineering import engineer_features
from data_cleaning import clean
from train import split_data, build_preprocessor

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Scores customers by churn risk so the retention team knows who to call first.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chart style ───────────────────────────────────────────────────────────────
BG      = "#0f1014"
SURFACE = "#16171d"
BORDER  = "#252630"
TEXT    = "#e8e6f0"
MUTED   = "#6b6b7e"
RED     = "#e05c5c"
GOLD    = "#c9a96e"
GREEN   = "#4fb87a"
BLUE    = "#378ADD"
ORANGE  = "#D85A30"
PURPLE  = "#7c6af7"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       BORDER,
    "grid.alpha":       0.6,
    "axes.grid":        True,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# ── Load artifacts once at startup ────────────────────────────────────────────

def load_artifacts():
    """Load model, preprocessor and threshold. Fails loudly if files are missing."""
    try:
        base = os.path.dirname(__file__)
        model_path        = os.path.join(base, "models", "churn_model.pkl")
        preprocessor_path = os.path.join(base, "models", "churn_preprocessor.pkl")
        threshold_path    = os.path.join(base, "models", "churn_threshold.pkl")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        with open(threshold_path, "rb") as f:
            threshold = pickle.load(f)
        return model, preprocessor, threshold
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found: {e}\n"
            "Run python run_pipeline.py first to train and save the model."
        )

model, preprocessor, threshold = load_artifacts()
print(f"Model loaded. Decision threshold: {threshold}")

# Cache test set predictions — loaded once, reused for all chart requests
_test_cache = {}

def get_test_data():
    """Load and cache test set so charts don't reload data on every request."""
    if _test_cache:
        return _test_cache

    base      = os.path.dirname(__file__)
    data_path = os.path.join(base, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df = clean(data_path)
    df = engineer_features(df)
    _, X_test, _, y_test = split_data(df)

    X_test_proc = preprocessor.transform(X_test)
    y_proba     = model.predict_proba(X_test_proc)[:, 1]
    y_pred      = (y_proba >= threshold).astype(int)

    _test_cache["X_test"]  = X_test
    _test_cache["y_test"]  = y_test
    _test_cache["y_proba"] = y_proba
    _test_cache["y_pred"]  = y_pred
    return _test_cache


def fig_to_response(fig) -> StreamingResponse:
    """Stream a matplotlib figure as a PNG — no file written to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")


# ── Schemas ───────────────────────────────────────────────────────────────────

class CustomerRecord(BaseModel):
    customerID:       str
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           int
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     float

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "9237-HQITU",
                "gender": "Female", "SeniorCitizen": 0,
                "Partner": "No", "Dependents": "No", "tenure": 2,
                "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70, "TotalCharges": 151.65
            }
        }


class ChurnPrediction(BaseModel):
    customerID:             str
    churn_probability:      float
    risk_level:             str
    recommendation:         str
    annual_revenue_at_risk: float


class BatchSummary(BaseModel):
    total_customers:       int
    high_risk_count:       int
    medium_risk_count:     int
    low_risk_count:        int
    total_revenue_at_risk: float
    predictions:           list[ChurnPrediction]


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_risk_level(probability: float) -> str:
    if probability >= config.RISK_BINS[2]:
        return "High"
    elif probability >= config.RISK_BINS[1]:
        return "Medium"
    return "Low"


def assign_recommendation(risk: str, contract: str, tenure: int) -> str:
    if risk == "High":
        if contract == "Month-to-month" and tenure <= 12:
            return "Priority call — offer discounted annual contract. High probability of leaving soon."
        elif contract == "Month-to-month":
            return "Priority call — long-tenure customer showing renewed risk. Loyalty offer recommended."
        return "Priority call — high churn risk. Escalate to retention team today."
    elif risk == "Medium":
        return "Monitor closely. Consider a proactive check-in or service upgrade offer."
    return "Low risk. No action needed — standard account management."


def score_dataframe(df: pd.DataFrame) -> list[ChurnPrediction]:
    """Core scoring — used by both single and batch endpoints."""
    df_feat = engineer_features(df.copy())
    X       = df_feat.drop(columns=["customerID", "Churn"], errors="ignore")
    X_proc  = preprocessor.transform(X)
    probs   = model.predict_proba(X_proc)[:, 1]

    results = []
    for i, row in df.iterrows():
        prob    = round(float(probs[i - df.index[0]]), 4)
        risk    = assign_risk_level(prob)
        rec     = assign_recommendation(risk, row["Contract"], row["tenure"])
        revenue = round(row["MonthlyCharges"] * 12, 2)
        results.append(ChurnPrediction(
            customerID=row["customerID"],
            churn_probability=prob,
            risk_level=risk,
            recommendation=rec,
            annual_revenue_at_risk=revenue
        ))
    return sorted(results, key=lambda x: x.churn_probability, reverse=True)


# ── Prediction routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return HTMLResponse("<h2>Frontend not found. Place index.html in /frontend/</h2>")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "Logistic Regression (class-weighted)",
            "threshold": threshold, "version": "1.0.0"}


@app.post("/predict", response_model=ChurnPrediction)
async def predict_single(customer: CustomerRecord):
    """Score one customer and return their risk profile."""
    try:
        df = pd.DataFrame([customer.model_dump()])
        return score_dataframe(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchSummary)
async def predict_batch(file: UploadFile = File(...)):
    """Score a full CSV and return a prioritised risk list."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        missing = [c for c in CustomerRecord.model_fields if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        predictions = score_dataframe(df)
        high   = [p for p in predictions if p.risk_level == "High"]
        medium = [p for p in predictions if p.risk_level == "Medium"]
        low    = [p for p in predictions if p.risk_level == "Low"]

        return BatchSummary(
            total_customers=len(predictions),
            high_risk_count=len(high),
            medium_risk_count=len(medium),
            low_risk_count=len(low),
            total_revenue_at_risk=round(sum(p.annual_revenue_at_risk for p in high), 2),
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    return {
        "model_type": "Logistic Regression",
        "class_weighting": "balanced",
        "decision_threshold": threshold,
        "imbalance_handling": "SMOTE-ENN",
        "priority_metric": "Recall",
        "recall": 0.890,
        "precision": 0.447,
        "roc_auc": 0.842,
        "missed_churners_on_test": 41,
        "risk_bands": {"Low": "0.0 – 0.3", "Medium": "0.3 – 0.6", "High": "0.6 – 1.0"},
        "top_churn_drivers": [
            "Month-to-month contract",
            "Tenure under 12 months",
            "Fiber optic with no add-on services",
            "Electronic check payment method"
        ],
        "note": (
            "Probabilities are relative risk scores for prioritisation, "
            "not literal churn rates. The model was trained on SMOTE-balanced "
            "data which inflates confidence — treat scores as rankings."
        )
    }


# ── Chart routes ──────────────────────────────────────────────────────────────

@app.get("/charts/risk-distribution")
async def chart_risk_distribution():
    """
    How many customers fall into each risk band.
    Gives the retention team an immediate sense of the workload.
    """
    data    = get_test_data()
    y_proba = data["y_proba"]

    counts = [
        int((y_proba >= config.RISK_BINS[2]).sum()),
        int(((y_proba >= config.RISK_BINS[1]) & (y_proba < config.RISK_BINS[2])).sum()),
        int((y_proba < config.RISK_BINS[1]).sum()),
    ]
    colors = [RED, GOLD, GREEN]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    bars = ax.bar(config.RISK_LABELS, counts, color=colors, edgecolor="none", width=0.5)
    ax.set_title("Customers by risk band", color=TEXT, pad=12)
    ax.set_ylabel("Number of customers", color=MUTED)
    ax.spines[:].set_color(BORDER)

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", color=TEXT, fontsize=12)

    plt.tight_layout()
    return fig_to_response(fig)


@app.get("/charts/revenue-at-risk")
async def chart_revenue_at_risk():
    """
    Annual revenue at risk per segment.
    Translates model output into dollar terms for business stakeholders.
    """
    data    = get_test_data()
    X_test  = data["X_test"]
    y_proba = data["y_proba"]
    annual  = X_test["MonthlyCharges"].values * 12

    revenues = [
        annual[y_proba >= config.RISK_BINS[2]].sum(),
        annual[(y_proba >= config.RISK_BINS[1]) & (y_proba < config.RISK_BINS[2])].sum(),
        annual[y_proba < config.RISK_BINS[1]].sum(),
    ]
    colors = [RED, GOLD, GREEN]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG)
    bars = ax.bar(config.RISK_LABELS, revenues, color=colors, edgecolor="none", width=0.5)
    ax.set_title("Annual revenue at risk by segment", color=TEXT, pad=12)
    ax.set_ylabel("Annual revenue ($)", color=MUTED)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.spines[:].set_color(BORDER)

    for bar, val in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"${val/1000:.0f}k", ha="center", va="bottom", color=TEXT, fontsize=11)

    plt.tight_layout()
    return fig_to_response(fig)


@app.get("/charts/top-churn-drivers")
async def chart_top_churn_drivers():
    """
    Churn rate by contract type, internet service and payment method.
    Shows what's actually driving customers to leave.
    """
    base      = os.path.dirname(__file__)
    data_path = os.path.join(base, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df        = clean(data_path)

    drivers = {
        "Contract":        "Contract type",
        "InternetService": "Internet service",
        "PaymentMethod":   "Payment method",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)

    for ax, (col, title) in zip(axes, drivers.items()):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=True) * 100
        bar_colors = [RED if v > 30 else BLUE for v in rates.values]
        bars = ax.barh(rates.index, rates.values, color=bar_colors, edgecolor="none")
        ax.set_title(title, color=TEXT, pad=10)
        ax.set_xlabel("Churn rate (%)", color=MUTED)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.spines[:].set_color(BORDER)

        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", color=TEXT, fontsize=9)

    plt.suptitle("Churn rate by key drivers", color=TEXT, fontsize=13, y=1.02)
    plt.tight_layout()
    return fig_to_response(fig)


@app.get("/charts/probability-distribution")
async def chart_probability_distribution():
    """
    Predicted probability split by actual churn label.
    Shows analysts how well the model separates churners from retained customers.
    """
    data    = get_test_data()
    y_proba = data["y_proba"]
    y_test  = data["y_test"].values

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.65,
            color=BLUE, label="Retained (actual)", edgecolor="none")
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.65,
            color=ORANGE, label="Churned (actual)", edgecolor="none")
    ax.axvline(x=threshold, color=PURPLE, linestyle="--",
               linewidth=2, label=f"Threshold ({threshold})")

    ax.set_title("Predicted probability distribution", color=TEXT, pad=12)
    ax.set_xlabel("Churn probability", color=MUTED)
    ax.set_ylabel("Count", color=MUTED)
    ax.legend(facecolor=SURFACE, labelcolor=TEXT, framealpha=0.9)
    ax.spines[:].set_color(BORDER)
    plt.tight_layout()
    return fig_to_response(fig)


@app.get("/charts/threshold-curve")
async def chart_threshold_curve():
    """
    Recall vs precision across decision thresholds.
    Helps analysts understand the cost of moving the threshold up or down.
    """
    from sklearn.metrics import recall_score, precision_score

    data    = get_test_data()
    y_proba = data["y_proba"]
    y_test  = data["y_test"]

    thresholds  = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    recalls, precisions, missed = [], [], []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        recalls.append(recall_score(y_test, y_pred_t))
        precisions.append(precision_score(y_test, y_pred_t))
        missed.append(int(((y_pred_t == 0) & (y_test == 1)).sum()))

    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor=BG)
    ax1.set_facecolor(SURFACE)
    ax1.plot(thresholds, recalls,    color=GREEN,  marker="o", linewidth=2, label="Recall")
    ax1.plot(thresholds, precisions, color=GOLD,   marker="o", linewidth=2, label="Precision")
    ax1.axvline(x=threshold, color=PURPLE, linestyle="--",
                linewidth=1.5, label=f"Chosen threshold ({threshold})")

    ax1.set_xlabel("Threshold", color=MUTED)
    ax1.set_ylabel("Score", color=MUTED)
    ax1.set_title("Recall vs Precision across thresholds", color=TEXT, pad=12)
    ax1.legend(facecolor=SURFACE, labelcolor=TEXT, framealpha=0.9)
    ax1.spines[:].set_color(BORDER)

    ax2 = ax1.twinx()
    ax2.bar(thresholds, missed, alpha=0.2, color=RED, width=0.03, label="Missed churners")
    ax2.set_ylabel("Missed churners", color=RED)
    ax2.tick_params(colors=RED)
    ax2.set_facecolor(SURFACE)
    ax2.spines[:].set_color(BORDER)

    plt.tight_layout()
    return fig_to_response(fig)

    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)