# Telco Customer Churn Prediction

Churn in telecom runs at 20–30% annually. In this dataset it was 26.5% —
roughly $1.5M in revenue exposure based on average monthly spend.
Most of those customers showed patterns before leaving. This project
builds a model to catch those patterns early enough to act on them.

**Live demo:** [your-app.streamlit.app](https://your-app.streamlit.app)
**API:** [your-app.up.railway.app](https://your-app.up.railway.app)


## What I Did

Started with a data quality issue standard checks missed — `TotalCharges`
was stored as a string, so blank values returned zero nulls. That kind of
silent error corrupts everything downstream if you don't catch it early.

EDA told a clear story before any modeling: month-to-month customers on
fiber optic paying by electronic check churned at nearly 3x everyone else.
Features were built around commitment signals — service depth, payment method,
and spend relative to tenure — not just raw service subscriptions.


## Model Selection

Three models trained on the same balanced data, evaluated on the same test set.
Logistic Regression won on recall — the metric that actually matters here.
Missing a churner costs far more than an unnecessary retention call.

| Model | Recall | Missed Churners | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.89 | 41 | 0.842 |
| GBM (RandomizedSearchCV) | 0.781 | 82 | 0.826 |
| Random Forest | 0.559 | — | 0.819 |

GBM was fully tuned with 30-iteration hyperparameter search — it still lost.
The gap between CV recall (0.968) and test recall (0.781) confirmed overfitting
driven by aggressive parameters on SMOTE-balanced folds.

Threshold tuning (0.5 → 0.35) with class weighting reduced missed churners
from 80 to 41, recovering an additional $30,420 per cycle over baseline.


## Key Findings

- Month-to-month contract is the single strongest churn predictor
- First 12 months is the highest-risk window — drops sharply after year one
- Fiber + month-to-month + no add-ons = highest combined risk profile
- Three distinct high-risk segments identified via K-Means clustering,
  each with a different recommended retention strategy


## What-If Analysis

The model was used to simulate retention interventions — not just flag customers.
A contract switch combined with one added service reduced a medium-high risk
customer's churn probability from 82% to 45%, with a retention break-even
point of $346. Below that cost, the intervention is profitable.


## How to Run It

**Install**
```bash
git clone https://github.com/GODFREY-PNG/telco-churn.git
cd telco-churn
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Start the API**
```bash
uvicorn main:app --reload
```
Runs at `http://127.0.0.1:8000` · Docs at `/docs`

**Start the dashboard**
```bash
streamlit run churn_app.py
```
Runs at `http://localhost:8501`


## API

**`POST /predict`** — scores one customer and returns their risk level and recommended action.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "9237-HQITU",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.70,
    "TotalCharges": 151.65
  }'
```

**`POST /predict/batch`** — upload a CSV and get back a full prioritised risk list.

**`GET /analytics`** — returns segment stats for the dashboard.

**`GET /charts/{chart_name}`** — returns chart images for the dashboard.

Available charts: `risk-distribution`, `revenue-at-risk`, `top-churn-drivers`,
`probability-distribution`, `threshold-curve`.


## Project Structure

```
TELCO_CHURN/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   ├── churn_preprocessor.pkl
│   └── churn_threshold.pkl
├── scripts/
│   ├── config.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── train.py
├── notebook/
│   └── CHURN_v2.ipynb
├── main.py
├── churn_app.py
├── run_pipeline.py
└── requirements.txt
```


## Stack

Python · pandas · scikit-learn · imbalanced-learn · SHAP · FastAPI · Uvicorn · Streamlit · matplotlib · seaborn


## On Metric Choice

A model predicting "no churn" for everyone scores 73% accuracy and catches
zero at-risk customers. Recall asks the right question — of everyone who
actually churned, how many did the model flag in time? That is the number
that connects to revenue.


## Honest Limitations

- Precision sits at 44.7% — about half the flagged customers will not actually churn.
  The model is intentionally tuned this way: a missed churner costs $777, a wasted
  retention call costs far less.
- The SMOTE-balanced training data inflates confidence scores. Treat probabilities
  as risk rankings for prioritisation, not literal churn rates.
- A `/retrain` endpoint would let the model update as new customer data comes in —
  not built yet but the pipeline structure supports it.