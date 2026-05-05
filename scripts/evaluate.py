"""
evaluate.py
Loads the saved model and reports performance on the test set.
Recall is the priority metric — missing a churner costs more than a false alarm.
All charts are saved to outputs/visualizations automatically.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)

import config
from data_cleaning import clean
from feature_engineering import engineer_features
from train import split_data, build_preprocessor

# All charts go here
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)


def save_fig(filename: str):
    """Save current figure to the visualizations folder and close it."""
    path = os.path.join(VIZ_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"Saved: outputs/visualizations/{filename}")


def load_artifacts():
    """Load model, preprocessor and threshold from disk."""
    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(config.THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)
    return model, preprocessor, threshold


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the metrics that matter for this business problem."""
    return {
        "Recall":            round(recall_score(y_true, y_pred), 3),
        "Precision":         round(precision_score(y_true, y_pred), 3),
        "F1":                round(f1_score(y_true, y_pred), 3),
        "ROC-AUC":           round(roc_auc_score(y_true, y_proba), 3),
        "Missed churners":   int(((y_pred == 0) & (y_true == 1)).sum())
    }


def plot_confusion_matrix(y_true, y_pred):
    """Confusion matrix — bottom-left is the most expensive cell."""
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix (threshold={config.THRESHOLD})", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    plt.tight_layout()
    save_fig("confusion_matrix.png")


def plot_threshold_curve(y_true, y_proba):
    """Shows how recall and precision change across thresholds."""
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    recalls, precisions, missed = [], [], []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred_t))
        precisions.append(precision_score(y_true, y_pred_t))
        missed.append(int(((y_pred_t == 0) & (y_true == 1)).sum()))

    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor="#0f0f0f")
    ax1.set_facecolor("#0f0f0f")

    ax1.plot(thresholds, recalls,    color="#5dbe8a", marker="o", label="Recall",    linewidth=2)
    ax1.plot(thresholds, precisions, color="#c9a96e", marker="o", label="Precision", linewidth=2)
    ax1.axvline(x=config.THRESHOLD, color="#7c6af7", linestyle="--", linewidth=1.5, label=f"Chosen threshold ({config.THRESHOLD})")

    ax1.set_xlabel("Threshold", color="white")
    ax1.set_ylabel("Score", color="white")
    ax1.set_title("Recall vs Precision across decision thresholds", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1a1a1a", labelcolor="white")
    ax1.spines[:].set_color("#2a2a35")

    # missed churners on secondary axis
    ax2 = ax1.twinx()
    ax2.bar(thresholds, missed, alpha=0.25, color="#e05c5c", width=0.03, label="Missed churners")
    ax2.set_ylabel("Missed churners", color="#e05c5c")
    ax2.tick_params(colors="#e05c5c")
    ax2.set_facecolor("#0f0f0f")

    plt.tight_layout()
    save_fig("threshold_curve.png")


def plot_roc_curve(y_true, y_proba):
    """ROC curve showing model discrimination ability."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    ax.plot(fpr, tpr, color="#7c6af7", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#2a2a35", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False positive rate", color="white")
    ax.set_ylabel("True positive rate", color="white")
    ax.set_title("ROC curve", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    ax.spines[:].set_color("#2a2a35")
    plt.tight_layout()
    save_fig("roc_curve.png")


def business_impact(y_true, y_pred, avg_monthly_charge=65):
    """Translate recall into dollar terms."""
    annual_rev = avg_monthly_charge * 12
    caught = int(((y_pred == 1) & (y_true == 1)).sum())
    missed = int(((y_pred == 0) & (y_true == 1)).sum())
    print(f"\nBusiness Impact (at ${avg_monthly_charge}/month avg):")
    print(f"  Caught churners : {caught} × ${annual_rev} = ${caught * annual_rev:,.0f} recoverable")
    print(f"  Missed churners : {missed} × ${annual_rev} = ${missed * annual_rev:,.0f} unrecoverable")


if __name__ == "__main__":
    df = clean(config.DATA_PATH)
    df = engineer_features(df)

    _, X_test, _, y_test = split_data(df)

    model, preprocessor, threshold = load_artifacts()
    X_test_proc = preprocessor.transform(X_test)

    y_proba = model.predict_proba(X_test_proc)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print("\nModel Performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["Retained", "Churned"]))
    business_impact(y_test, y_pred)

    plot_confusion_matrix(y_test, y_pred)
    plot_threshold_curve(y_test, y_proba)
    plot_roc_curve(y_test, y_proba)

    print("\nAll evaluation charts saved to outputs/visualizations/")