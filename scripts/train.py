"""
train.py
Splits data, preprocesses, balances with SMOTE-ENN, trains the
production model and saves model + preprocessor + threshold to disk.
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN

import config
from data_cleaning import clean
from feature_engineering import engineer_features


def split_data(df: pd.DataFrame):
    """Stratified split — keeps churn ratio consistent across train and test."""
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Scale numeric features, one-hot encode categoricals.
    Fitted on training data only — prevents data leakage.
    """
    cat_cols = X_train.select_dtypes("object").columns.tolist()
    num_cols = X_train.select_dtypes("number").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    return preprocessor


def balance_data(X_train, y_train):
    """
    SMOTE-ENN: oversample minority class then clean borderline samples.
    Applied only to training data — test set stays untouched.
    """
    sampler = SMOTEENN(random_state=config.RANDOM_STATE)
    X_bal, y_bal = sampler.fit_resample(X_train, y_train)
    print(f"After SMOTE-ENN — Churn: {y_bal.sum():,} | No churn: {(y_bal == 0).sum():,}")
    return X_bal, y_bal


def train_model(X_train, y_train) -> LogisticRegression:
    """
    Class-weighted LR penalises missed churners more heavily.
    Missing a churner costs far more than a false alarm.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=config.RANDOM_STATE
    )
    model.fit(X_train, y_train)
    print("Model trained: class-weighted Logistic Regression")
    return model


def save_artifacts(model, preprocessor):
    """Save model, preprocessor and threshold — all three needed at inference time."""
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(config.PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(config.THRESHOLD_PATH, "wb") as f:
        pickle.dump(config.THRESHOLD, f)
    print(f"Saved: {config.MODEL_PATH}")
    print(f"Saved: {config.PREPROCESSOR_PATH}")
    print(f"Saved: {config.THRESHOLD_PATH}")


if __name__ == "__main__":
    df = clean(config.DATA_PATH)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    X_train_bal, y_train_bal = balance_data(X_train_proc, y_train)

    model = train_model(X_train_bal, y_train_bal)

    save_artifacts(model, preprocessor)