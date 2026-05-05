"""
feature_engineering.py
Builds model ready features from the cleaned dataframe.
Each feature is grounded in what EDA revealed about churn drivers.
"""

import pandas as pd
import config


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many add-on services a customer has.
    More add-ons = deeper tie-in = harder to switch.
    """
    df["total_services"] = df[config.SERVICE_COLS].apply(
        lambda col: col.str.contains("Yes", na=False)
    ).sum(axis=1)
    return df


def add_contract_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Month-to-month is the single strongest churn signal from EDA."""
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_charge_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spend relative to tenure — high ratio means customer hasn't seen value yet.
    Cap at 99th percentile to prevent tenure=0 from creating extreme values.
    """
    df["charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    cap = df["charge_per_month"].quantile(0.99)
    df["charge_per_month"] = df["charge_per_month"].clip(upper=cap)
    return df


def add_payment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Electronic check correlates with lowest customer commitment in EDA."""
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def add_fiber_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Fiber customers pay more and have more alternatives to compare."""
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    return df


def add_interaction_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiber × month-to-month interaction — premium price with zero commitment.
    Linear models can't capture this combination automatically,
    so we create it explicitly.
    """
    df["fiber_x_mtm"] = df["is_fiber"] * df["is_month_to_month"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = add_service_count(df)
    df = add_contract_flag(df)
    df = add_charge_per_month(df)
    df = add_payment_flag(df)
    df = add_fiber_flag(df)
    df = add_interaction_feature(df)

    # tenure_band was only for EDA charts — not needed in model training
    if "tenure_band" in df.columns:
        df = df.drop(columns=["tenure_band"])

    print(f"Features added: total_services, is_month_to_month, charge_per_month, "
          f"is_electronic_check, is_fiber, fiber_x_mtm")
    return df


if __name__ == "__main__":
    from data_cleaning import clean
    df = clean(config.DATA_PATH)
    df = engineer_features(df)
    print(df[["total_services", "is_month_to_month", "charge_per_month",
              "is_electronic_check", "is_fiber", "fiber_x_mtm"]].describe())