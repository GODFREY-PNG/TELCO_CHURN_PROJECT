"""
visualize.py
Generates and saves all EDA and model charts to outputs/visualizations/.
Run this after training to get the full visual story of the data and model.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

import config
from data_cleaning import clean
from feature_engineering import engineer_features
from train import split_data, build_preprocessor, balance_data, train_model

# Output folder
VIZ_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "visualizations")
os.makedirs(VIZ_DIR, exist_ok=True)

# Chart style — consistent dark theme across all plots
plt.rcParams.update({
    "figure.facecolor":  "#0f0f0f",
    "axes.facecolor":    "#16161c",
    "axes.edgecolor":    "#2a2a35",
    "axes.labelcolor":   "#aaaaaa",
    "xtick.color":       "#aaaaaa",
    "ytick.color":       "#aaaaaa",
    "text.color":        "#e8e6f0",
    "grid.color":        "#2a2a35",
    "grid.alpha":        0.5,
    "axes.grid":         True,
    "font.family":       "sans-serif",
})

COLORS = {
    "blue":   "#378ADD",
    "orange": "#D85A30",
    "green":  "#5dbe8a",
    "gold":   "#c9a96e",
    "purple": "#7c6af7",
    "red":    "#e05c5c",
    "muted":  "#6b6b7e",
}


def save_fig(filename: str):
    """Save current figure and close cleanly."""
    path = os.path.join(VIZ_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close()
    print(f"  Saved: {filename}")


# ── EDA CHARTS ────────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame):
    """Class imbalance — sets context for why we use recall over accuracy."""
    churn_pct = df["Churn"].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Retained", "Churned"],
        churn_pct.values,
        color=[COLORS["blue"], COLORS["orange"]],
        width=0.5, edgecolor="none"
    )
    ax.set_title("Churn distribution", pad=12)
    ax.set_ylabel("% of customers")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, churn_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    save_fig("01_churn_distribution.png")


def plot_churn_by_category(df: pd.DataFrame):
    """Churn rate by contract, internet service and payment method."""
    cols = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, cols):
        rates = df.groupby(col)["Churn"].mean().sort_values(ascending=False) * 100
        bars = ax.barh(rates.index, rates.values,
                       color=[COLORS["orange"] if v > 30 else COLORS["blue"] for v in rates.values],
                       edgecolor="none")
        ax.set_title(col, pad=10)
        ax.set_xlabel("Churn rate (%)")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9)

    plt.suptitle("Churn rate by key categorical features", y=1.02, fontsize=13)
    plt.tight_layout()
    save_fig("02_churn_by_category.png")


def plot_numeric_distributions(df: pd.DataFrame):
    """KDE distributions of numeric features split by churn status."""
    cols = {
        "tenure":         "Months with company",
        "MonthlyCharges": "Monthly bill (USD)",
        "TotalCharges":   "Total paid (USD)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, label) in zip(axes, cols.items()):
        for churn_val, color, name in [(0, COLORS["blue"], "Retained"), (1, COLORS["orange"], "Churned")]:
            subset = df[df["Churn"] == churn_val][col].dropna()
            subset.plot.kde(ax=ax, color=color, label=name, linewidth=2)

        ax.set_title(label, pad=10)
        ax.set_xlabel("")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

    plt.suptitle("Distribution of numeric features by churn status", y=1.02, fontsize=13)
    plt.tight_layout()
    save_fig("03_numeric_distributions.png")


def plot_tenure_churn_rate(df: pd.DataFrame):
    """Churn rate by tenure band — shows the first-year risk clearly."""
    df = df.copy()
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    )
    rates = df.groupby("tenure_band", observed=True)["Churn"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rates.index, rates.values,
                  color=[COLORS["orange"] if v > 25 else COLORS["blue"] for v in rates.values],
                  edgecolor="none", width=0.5)
    ax.set_title("Churn rate by tenure band", pad=12)
    ax.set_ylabel("Churn rate (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    save_fig("04_tenure_churn_rate.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    num_features = df.select_dtypes("number").drop(columns=["Churn"], errors="ignore")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        num_features.corr(),
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="#0f0f0f",
        ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Feature correlation heatmap", pad=12)
    plt.tight_layout()
    save_fig("05_correlation_heatmap.png")


# ── MODEL CHARTS ──────────────────────────────────────────────────────────────

def plot_odds_ratios(model: LogisticRegression, all_features: list):
    """Odds ratios from LR coefficients — shows directional churn drivers."""
    odds_ratios = pd.Series(
        np.exp(model.coef_[0]), index=all_features
    ).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    odds_ratios.tail(12).plot(kind="barh", ax=axes[0], color=COLORS["orange"], edgecolor="none")
    axes[0].axvline(x=1, color=COLORS["blue"], linestyle="--", linewidth=1.5, label="No effect")
    axes[0].set_title("Features increasing churn risk", pad=10)
    axes[0].set_xlabel("Odds ratio")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    odds_ratios.head(12).plot(kind="barh", ax=axes[1], color=COLORS["blue"], edgecolor="none")
    axes[1].axvline(x=1, color=COLORS["orange"], linestyle="--", linewidth=1.5, label="No effect")
    axes[1].set_title("Features reducing churn risk", pad=10)
    axes[1].set_xlabel("Odds ratio")
    axes[1].legend(facecolor="#1a1a1a", labelcolor="white")

    plt.suptitle("Logistic Regression — odds ratios", y=1.02, fontsize=13)
    plt.tight_layout()
    save_fig("06_odds_ratios.png")


def plot_before_after_confusion(y_test, y_pred_baseline, y_pred_tuned):
    """Side-by-side confusion matrix showing improvement from threshold tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_baseline,
        display_labels=["Retained", "Churned"],
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Before tuning (threshold=0.5)", color="white")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_tuned,
        display_labels=["Retained", "Churned"],
        cmap="Oranges", ax=axes[1]
    )
    axes[1].set_title(f"After tuning (threshold={config.THRESHOLD})", color="white")

    for ax in axes:
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    plt.suptitle("Confusion matrix — threshold tuning impact", y=1.02, fontsize=13)
    plt.tight_layout()
    save_fig("07_confusion_before_after.png")


def plot_risk_distribution(churn_scores: pd.DataFrame):
    """Distribution of customers across risk bands."""
    counts = churn_scores["risk_level"].value_counts().reindex(["High", "Medium", "Low"])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLORS["red"], COLORS["gold"], COLORS["green"]]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.5)

    ax.set_title("Customer count by risk band", pad=12)
    ax.set_ylabel("Number of customers")

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    save_fig("08_risk_distribution.png")


def plot_probability_histogram(y_proba, y_test):
    """Predicted probability distribution split by actual churn label."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(y_proba[y_test == 0], bins=40, alpha=0.6,
            color=COLORS["blue"], label="Retained (actual)", edgecolor="none")
    ax.hist(y_proba[y_test == 1], bins=40, alpha=0.6,
            color=COLORS["orange"], label="Churned (actual)", edgecolor="none")
    ax.axvline(x=config.THRESHOLD, color=COLORS["purple"], linestyle="--",
               linewidth=2, label=f"Threshold ({config.THRESHOLD})")

    ax.set_title("Predicted churn probability distribution", pad=12)
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Count")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")

    plt.tight_layout()
    save_fig("09_probability_histogram.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all visualizations...\n")

    df = clean(config.DATA_PATH)
    df_feat = engineer_features(df.copy())

    X_train, X_test, y_train, y_test = split_data(df_feat)
    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    X_train_bal, y_train_bal = balance_data(X_train_proc, y_train)
    model = train_model(X_train_bal, y_train_bal)

    y_proba        = model.predict_proba(X_test_proc)[:, 1]
    y_pred_baseline = (y_proba >= 0.5).astype(int)
    y_pred_tuned    = (y_proba >= config.THRESHOLD).astype(int)

    # Feature names after encoding
    cat_cols     = X_train.select_dtypes("object").columns.tolist()
    num_cols     = X_train.select_dtypes("number").columns.tolist()
    cat_names    = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_features = num_cols + cat_names

    # Risk scores for distribution plot
    churn_scores = pd.DataFrame({
        "churn_probability": y_proba,
        "risk_level": pd.cut(
            y_proba,
            bins=config.RISK_BINS,
            labels=config.RISK_LABELS
        )
    })

    print("EDA charts:")
    plot_churn_distribution(df_feat)
    plot_churn_by_category(df_feat)
    plot_numeric_distributions(df_feat)
    plot_tenure_churn_rate(df_feat)
    plot_correlation_heatmap(df_feat)

    print("\nModel charts:")
    plot_odds_ratios(model, all_features)
    plot_before_after_confusion(y_test, y_pred_baseline, y_pred_tuned)
    plot_risk_distribution(churn_scores)
    plot_probability_histogram(y_proba, y_test.values)

    print(f"\nAll charts saved to outputs/visualizations/")
    print(f"Total: 9 charts")