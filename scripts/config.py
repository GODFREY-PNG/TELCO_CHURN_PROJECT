"""
Central configuration — all settings live here.
Change values here and every script picks them up automatically.
"""

# Paths
DATA_PATH         = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH        = "../models/churn_model.pkl"
PREPROCESSOR_PATH = "../models/churn_preprocessor.pkl"
THRESHOLD_PATH    = "../models/churn_threshold.pkl"
VIZ_DIR           = "../outputs/visualizations"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Model decision threshold — tuned for recall over precision
THRESHOLD = 0.35

# Risk scoring bands
RISK_BINS   = [0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]

# Service columns used in feature engineering
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]