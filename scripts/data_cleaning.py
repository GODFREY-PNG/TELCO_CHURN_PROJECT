"""
data_cleaning.py
Loads raw data, fixes known issues, and returns a clean dataframe.
"""

import pandas as pd
import config


def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV from disk."""
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is stored as string — blank values read as valid entries.
    Coercing to numeric surfaces them, then imputes with median.
    Rows with tenure=0 have no charges posted yet, so dropping them
    would remove real customers for a data timing issue.
    """
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    hidden_nulls = df["TotalCharges"].isnull().sum()
    print(f"Hidden blanks found in TotalCharges: {hidden_nulls}")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate customer IDs — a duplicate teaches the model the same customer twice."""
    dupes = df.duplicated(subset="customerID").sum()
    if dupes > 0:
        df = df.drop_duplicates(subset="customerID")
        print(f"Removed {dupes} duplicate customer IDs")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn column from Yes/No to 1/0."""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def clean(path: str) -> pd.DataFrame:
    """Full cleaning pipeline — load, fix, deduplicate, encode."""
    df = load_data(path)
    df = fix_total_charges(df)
    df = remove_duplicates(df)
    df = encode_target(df)
    print(f"Clean data ready: {df.shape[0]:,} rows, {df.isnull().sum().sum()} nulls remaining")
    return df


if __name__ == "__main__":
    df = clean(config.DATA_PATH)
    print(df.head())