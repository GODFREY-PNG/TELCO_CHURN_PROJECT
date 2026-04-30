# Telco Customer Churn Prediction

Churn in telecom sits around 20–30% annually. In this dataset it was 26.5% —
which works out to roughly $1.5M in revenue exposure based on average monthly spend.
Most of those customers didn't just disappear. They showed patterns weeks before leaving.

The question I tried to  answer : can those patterns be caught early enough
to actually do something about it?

## What I Focused On

The first thing I noticed was a data quality issue that standard checks missed.
`TotalCharges` was stored as a string — blank values looked like valid entries,
so `.isnull()` returned zero. That kind of silent error corrupts everything downstream
if you don't catch it before modeling.

Once the data was clean, the EDA told a clear story. Three things stood out:
contract type, internet service, and payment method explained most of the churn
before a single model was trained. Month-to-month customers on fiber optic
paying by electronic check were churning at nearly 3x the rate of everyone else.

I built features around that — not just what services a customer had,
but how committed they looked. Service depth, payment method, and spend
relative to tenure ended up being the most useful signals.



## Why I Chose Logistic Regression

I trained three models. Gradient Boosting had the best accuracy.
Logistic Regression had the best recall — and recall is the metric
that actually matters here.

Missing a churner means losing that customer entirely.
A false alarm means one unnecessary retention call.
Those two mistakes are not the same cost, so optimising for accuracy
would have been the wrong call.

| Model | Recall | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.789 | 0.846 |
| Gradient Boosting | 0.620 | 0.836 |
| Random Forest | 0.559 | 0.819 |

Even then, the default threshold of 0.5 was still missing 80 customers
who actually churned. I combined class weighting with threshold tuning —
dropping the decision boundary to 0.35 — which pushed recall higher
while keeping precision at a level the retention team could work with.



## What the Model Actually Found

- Month-to-month contract was the single strongest churn signal
- Customers in their first 12 months churned at nearly 3x the long-term rate
- Fiber optic users without any support add-ons kept appearing at the top of the risk list
- Electronic check payers showed the lowest payment commitment across all segments

The output isn't just a metric — it's a ranked list of customers by churn probability
with a risk label attached. Something a retention team can open on a Monday morning
and start calling from the top.



## Stack

Python · pandas · scikit-learn · imbalanced-learn · SHAP · matplotlib



## Project Structure
├── notebook/
│   └── CHURN.ipynb
├── models/
│   └── churn_model.pkl
│   └── churn_preprocessor.pkl
│   └── churn_threshold.pkl
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
Scripts and API endpoint coming soon.



## A Note on the Metric Choice

73% of customers in this dataset did not churn.
A model that predicts "no churn" for everyone scores 73% accuracy
and catches exactly zero customers at risk.

Recall asks the right question — of everyone who actually churned,
how many did the model flag before they left?

That is the number that connects to revenue.