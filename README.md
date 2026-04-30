# Telco Customer Churn Prediction

Telecom companies lose 20–30% of customers every year to churn.
Most of those customers showed warning signs before they left —
this project builds a model to catch them early.


## The Problem

Using this dataset, the churn rate is 26.5% across 7,043 customers.
At an average monthly bill of ~$65, that is roughly **$1.5M in annual
revenue walking out the door** — much of it preventable with early action.

The goal is not just to predict churn. It is to give the retention team
a ranked list of customers to call before they cancel.



## What This Project Covers

- Identified a hidden data issue — `TotalCharges` stored as string,
  masking blank values that standard null checks missed entirely
- Explored which customer segments churn most — contract type,
  internet service, and payment method tell most of the story
- Built features that capture customer behaviour, not just subscriptions —
  service depth, payment commitment, and value perception
- Trained three models and selected based on **recall**, not accuracy —
  missing a churner costs more than a false alarm
- Reduced missed churners by combining class weighting with threshold
  tuning — moving the decision boundary from 0.5 to 0.35
- Produced a ranked list of high-risk customers with churn probability
  scores the retention team can act on directly



## Results

| Model | Recall | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.789 | 0.846 |
| Random Forest | 0.559 | 0.819 |
| Gradient Boosting | 0.620 | 0.836 |

**Logistic Regression** performed best on recall — the metric that
matters most when every missed churner represents lost revenue.

After threshold tuning, missed churners dropped from 80 toward a lower
number while keeping precision at a level the team can work with.



## Top Churn Signals Found

- Month-to-month contract — strongest predictor by a wide margin
- Tenure under 12 months — first year is the highest-risk window
- Fiber optic without support add-ons — high bill, low perceived value
- Electronic check payment — lowest payment commitment in the dataset


## Stack

- Python · pandas · scikit-learn · imbalanced-learn · SHAP · matplotlib



## Project Structure
├── notebook/
│   └── CHURN.ipynb          # full analysis notebook
├── models/
│   └── churn_model.pkl      # trained logistic regression
│   └── churn_preprocessor.pkl
│   └── churn_threshold.pkl  # tuned decision threshold
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
*Scripts and API coming soon.*


## Why Recall Over Accuracy

Accuracy on this dataset is misleading — predicting "no churn" every time
scores 73% without catching a single customer at risk.

Recall answers the right question: **of all customers who actually churned,
how many did the model flag in time?**

A false alarm costs one unnecessary retention call.
A missed churner costs the full customer — plus 5× the acquisition cost
to find a replacement.

