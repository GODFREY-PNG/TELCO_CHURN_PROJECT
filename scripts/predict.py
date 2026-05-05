"""
predict.py
Scores new customer records and returns churn probability + risk level.
This is what a FastAPI endpoint would call internally.
"""

import pickle
import pandas as pd

import config
from feature_engineering import engineer_features


def load_artifacts():
    """Load model, preprocessor and threshold from disk."""
    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(config.PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    with open(config.THRESHOLD_PATH, "rb") as f:
        threshold = pickle.load(f)
    return model, preprocessor, threshold


def assign_risk_level(probability: float) -> str:
    """Maping churn probability to a human-readable risk label."""
    if probability >= 0.6:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    return "Low"


def predict(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw customer records (as they come from the source system),
    runs the full pipeline and returns churn scores.

    Input:  raw dataframe with original column names
    Output: same dataframe with churn_probability and risk_level added
    """
    model, preprocessor, threshold = load_artifacts()

    # Apply same feature engineering as training
    df = engineer_features(raw_df.copy())

    # Drop columns not used during training
    customer_ids = df["customerID"].copy()
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    X_proc = preprocessor.transform(X)
    probabilities = model.predict_proba(X_proc)[:, 1]

    results = raw_df[["customerID"]].copy()
    results["churn_probability"] = probabilities.round(4)
    results["risk_level"] = results["churn_probability"].apply(assign_risk_level)
    results = results.sort_values("churn_probability", ascending=False)

    return results


if __name__ == "__main__":
    # Quick test — score a sample of raw records
    from data_cleaning import clean

    df_raw = clean(config.DATA_PATH)

    # Score first 10 customers as a sanity check
    sample = df_raw.head(10)
    scores = predict(sample)
    print(scores.to_string(index=False))