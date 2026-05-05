import streamlit as st
import requests
import pandas as pd

# replace with your actual railway url before deploying
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Churn Intelligence — Telco Retention",
    page_icon="📡",
    layout="wide"
)

# same card and insight styling as the segmentation app
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1100px;
        margin: auto;
    }

    @media (max-width: 768px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .section-card { padding: 1rem 1.1rem !important; }
        .insight-strip { padding: 0.75rem 0.9rem !important; }
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }

    .insight-strip {
        border-left: 5px solid;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        background: #fafafa;
        margin-bottom: 1rem;
    }

    .label-tag {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 5px;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .footer-note {
        font-size: 0.72rem;
        color: #9ca3af;
        text-align: center;
        padding-top: 1rem;
    }

    @media (max-width: 480px) {
        [data-testid="metric-container"] { font-size: 0.85rem; }
    }
</style>
""", unsafe_allow_html=True)


# check if the api is live
def api_is_up():
    try:
        r = requests.get(f"{API_URL}/health", timeout=4)
        return r.status_code == 200
    except Exception:
        return False


# score one customer against the api
def call_predict(payload):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Prediction failed. ({e})")
        return None


# send a full csv to the batch endpoint
def call_batch(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_URL}/predict/batch",
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Batch scoring failed. ({e})")
        return None


# fetch model info once and cache it
@st.cache_data(ttl=300)
def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not load model info. ({e})")
        return None


# fetch a chart image from the api
def get_chart(endpoint):
    try:
        r = requests.get(f"{API_URL}/charts/{endpoint}", timeout=15)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None


# page header
st.markdown("## 📡 Churn Intelligence — Telco Retention")

# show api status clearly so the user knows immediately if something is wrong
if api_is_up():
    st.success("API connected")
else:
    st.warning("API offline. Start the FastAPI server first: `uvicorn main:app --reload`")

st.markdown(
    "Built on Logistic Regression trained on **7,043 Telco customers** "
    "across **20 account and service features**: contract type, tenure, monthly charges, "
    "internet service, payment method, phone service, online security, online backup, "
    "device protection, tech support, streaming TV, streaming movies, paperless billing, "
    "multiple lines, senior citizen status, partner, dependents, gender, and total charges."
)

st.divider()

# why this exists — plain language problem statement
st.markdown("""
<div class="section-card">
<h4>Why this exists</h4>
<p>
Telco companies lose customers every month and most of the time they find out too late.
By the time a customer calls to cancel, the decision is usually already made.
</p>
<p>
This tool scores every customer by their likelihood of leaving before they do.
The retention team gets a prioritised list — who to call first, why they are at risk,
and exactly what to offer. No more guessing or treating every customer the same
when the risk levels are completely different.
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# four tabs matching the segmentation app structure
tab1, tab2, tab3, tab4 = st.tabs([
    "**Overview**",
    "**Score a Customer**",
    "**Batch Scoring**",
    "**Model Details**"
])


# OVERVIEW TAB
# all numbers here come directly from the notebook — no estimates
with tab1:
    st.markdown("### What the data shows")

    # three headline cards using real notebook numbers
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
<div class="section-card" style="border-top: 4px solid #dc2626;">
<b style="color:#dc2626">CHURN RATE</b><br><br>
<span style="font-size:2rem; font-weight:700">26.5%</span>
<span style="color:#6b7280"> of customers leave</span><br>
<span style="font-size:1.4rem; font-weight:700">1,869</span>
<span style="color:#6b7280"> out of 7,043 churned</span><br><br>
<span style="color:#374151">More than 1 in 4 customers is leaving.
The model catches 89 out of every 100 of them before they go.</span>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="section-card" style="border-top: 4px solid #f59e0b;">
<b style="color:#f59e0b">REVENUE AT RISK</b><br><br>
<span style="font-size:2rem; font-weight:700">$1,452,475</span>
<span style="color:#6b7280"> total at risk</span><br>
<span style="font-size:1.4rem; font-weight:700">$777</span>
<span style="color:#6b7280"> avg annual value per customer</span><br><br>
<span style="color:#374151">Every churner that leaves without a retention attempt
is $777 in annual revenue that did not have to go.</span>
</div>
""", unsafe_allow_html=True)

    with col3:
        st.markdown("""
<div class="section-card" style="border-top: 4px solid #10b981;">
<b style="color:#10b981">HIGHEST RISK GROUP</b><br><br>
<span style="font-size:2rem; font-weight:700">2,128</span>
<span style="color:#6b7280"> fiber + month-to-month customers</span><br>
<span style="font-size:1.4rem; font-weight:700">54.6%</span>
<span style="color:#6b7280"> churn rate in this group</span><br><br>
<span style="color:#374151">Fiber optic customers on month-to-month contracts
churn at more than 3x the rate of everyone else. This group needs immediate attention.</span>
</div>
""", unsafe_allow_html=True)

    # the single most important business finding
    st.markdown("""
<div class="section-card" style="border-left: 5px solid #dc2626; background: #fef2f2;">
<h4 style="color:#991b1b">The cost of doing nothing</h4>
<p style="font-size:1rem; color:#374151">
<b>$1,452,475 in annual revenue is sitting in customers who are actively at risk of leaving.</b>
The notebook shows that a single contract-switch offer on one high-risk customer
can reduce their churn probability from 82% to 45% — an expected retention value of $347.
Multiply that across the 604 high-risk customers flagged in testing and the return on a
structured retention programme is clear.
</p>
</div>
""", unsafe_allow_html=True)

    # what each risk level means for the team
    st.markdown("### What the risk levels mean")
    st.markdown("Every customer gets one of three risk labels based on their churn probability.")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("""
<div class="section-card" style="border-left: 5px solid #dc2626;">
<b style="color:#dc2626">HIGH RISK — 60% or above</b><br><br>
<span style="color:#374151">Call this week. Offer a contract upgrade or a discount
on an annual plan. In testing, 604 customers were flagged here.
Every day without contact increases the chance the decision is already made.</span>
</div>
""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
<div class="section-card" style="border-left: 5px solid #f59e0b;">
<b style="color:#f59e0b">MEDIUM RISK — 30% to 60%</b><br><br>
<span style="color:#374151">Monitor and check in. A proactive service upgrade
or loyalty offer at the right moment keeps these customers stable.
Some will tip into high risk without any contact.</span>
</div>
""", unsafe_allow_html=True)

    with r3:
        st.markdown("""
<div class="section-card" style="border-left: 5px solid #10b981;">
<b style="color:#10b981">LOW RISK — under 30%</b><br><br>
<span style="color:#374151">Standard account management. No urgent action needed.
Keep retention budget focused on the two groups above
where every call has a measurable return.</span>
</div>
""", unsafe_allow_html=True)

    # key insights — all numbers from the notebook
    st.markdown("### What drives customers to leave")
    st.caption("All figures are from the training data — 7,043 Telco customers.")

    insights = [
        {
            "type": "RISK",
            "color": "#dc2626",
            "title": "Fiber optic customers on month-to-month contracts churn at 54.6%",
            "detail": (
                "2,128 customers sit in this group — fiber internet with no contract commitment. "
                "Their churn rate is 54.6% versus 14.4% for everyone else. "
                "They are paying the highest monthly charges with nothing keeping them from leaving tomorrow."
            )
        },
        {
            "type": "RISK",
            "color": "#dc2626",
            "title": "New customers churn before they find value",
            "detail": (
                "The highest-risk customers in testing had tenures of 1 to 2 months. "
                "The first year is where the relationship is built or lost. "
                "Early onboarding contact and a contract incentive in month one can shift a customer "
                "from 99% churn probability to long-term retained."
            )
        },
        {
            "type": "REVENUE",
            "color": "#10b981",
            "title": "One retention conversation is worth $347 per high-risk customer",
            "detail": (
                "The notebook models a real intervention: switching a high-risk customer "
                "(82% churn probability) from month-to-month to an annual contract plus one added service "
                "drops their probability to 45%. "
                "At $777 annual value, the expected retention return is $347. "
                "Any retention offer costing less than that is profitable."
            )
        },
        {
            "type": "OPPORTUNITY",
            "color": "#1d4ed8",
            "title": "Average monthly charge is $64.76 — fibre customers pay significantly more",
            "detail": (
                "The average monthly charge across all 7,043 customers is $64.76, giving an estimated "
                "annual value of $777 per customer. "
                "Fiber optic customers sit well above this average, which means losing them costs more "
                "and retaining them returns more than the headline figure suggests."
            )
        },
    ]

    for ins in insights:
        st.markdown(
            f"<div class='insight-strip' style='border-color:{ins['color']};'>"
            f"<span style='color:{ins['color']}; font-size:0.7rem; font-weight:700'>{ins['type']}</span><br>"
            f"<b style='font-size:1rem'>{ins['title']}</b><br>"
            f"<span style='color:#374151'>{ins['detail']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # charts from the api — only show if the api is reachable
    st.markdown("### Model charts")
    st.caption("Live from the API — showing test set results.")

    chart_list = [
        ("risk-distribution",       "Customers by risk band",           "How many customers fall into each risk level."),
        ("revenue-at-risk",         "Revenue at risk by segment",        "What annual revenue sits in each risk band."),
        ("top-churn-drivers",       "Top churn drivers",                 "Which account features drive the most churn."),
        ("probability-distribution","Probability distribution",          "How well the model separates churners from retained customers."),
        ("threshold-curve",         "Recall vs Precision curve",         "What happens to recall and precision as the decision threshold changes."),
    ]

    # show charts in pairs so the page does not get too long
    for i in range(0, len(chart_list), 2):
        pair = chart_list[i:i+2]
        cols = st.columns(len(pair), gap="large")
        for col, (endpoint, title, desc) in zip(cols, pair):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)
                img = get_chart(endpoint)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.info("Chart not available — API must be running locally to load charts.")

    # model details tucked away — not the main story for stakeholders
    with st.expander("Model details (for technical reviewers)"):
        st.markdown("""
- **Algorithm:** Logistic Regression with balanced class weighting
- **Training records:** 7,043 Telco customers
- **Imbalance handling:** SMOTE-ENN (synthetic oversampling + cleaning)
- **Priority metric:** Recall — missing a churner costs more than a false alarm
- **Recall:** 89% at threshold 0.35 — catches 333 out of 374 churners in the test set
- **Precision:** 44.7%
- **ROC-AUC:** 0.842
- **Decision threshold:** 0.35 (lowered from 0.5 to prioritise recall over precision)
- **Missed churners on test set:** 41 out of 374
- **Note:** Probabilities are relative risk rankings, not literal churn rates. Treat them as a prioritisation tool, not a guarantee.
""")


# SCORE A CUSTOMER TAB
# single customer prediction
with tab2:
    st.markdown("### Score a single customer")
    st.markdown(
        "Fill in the customer's account details and click **Run Prediction**. "
        "The model returns their churn risk, revenue at stake, and what to do."
    )

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3, gap="large")

    with col_a:
        st.markdown("**Account details**")
        customer_id    = st.text_input("Customer ID", value="9237-HQITU")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner        = st.selectbox("Partner", ["Yes", "No"])
        dependents     = st.selectbox("Dependents", ["Yes", "No"])
        tenure         = st.slider("Tenure (months)", 0, 72, 2)

    with col_b:
        st.markdown("**Services**")
        phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup    = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protect   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support     = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col_c:
        st.markdown("**Billing**")
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", value=70.70, step=0.01, min_value=0.0)
        total_charges   = st.number_input("Total Charges ($)", value=151.65, step=0.01, min_value=0.0)

    st.markdown("---")
    run = st.button("Run Prediction", type="primary", use_container_width=True)

    if run:
        payload = {
            "customerID":       customer_id,
            "gender":           gender,
            "SeniorCitizen":    senior_citizen,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protect,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment_method,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        with st.spinner("Running model..."):
            result = call_predict(payload)

        if result:
            risk = result["risk_level"]
            risk_colors = {"High": "#dc2626", "Medium": "#f59e0b", "Low": "#10b981"}
            risk_icons  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            color = risk_colors.get(risk, "#6b7280")
            icon  = risk_icons.get(risk, "⚪")

            st.markdown(f"""
<div class="section-card" style="border-left: 5px solid {color}; background: {color}0d;">
<h3 style="margin:0">{icon} {risk} Risk — {result['customerID']}</h3>
</div>
""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            m2.metric("Risk Level", risk)
            m3.metric("Annual Revenue at Risk", f"${result['annual_revenue_at_risk']:,.0f}")

            st.markdown("---")
            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("**Key account signals**")
                st.markdown(f"""
<div class="section-card">
<p style="color:#374151; line-height:1.8">
Contract: <b>{contract}</b><br>
Tenure: <b>{tenure} months</b><br>
Internet service: <b>{internet_service}</b><br>
Payment method: <b>{payment_method}</b><br>
Monthly charges: <b>${monthly_charges:,.2f}</b>
</p>
</div>
""", unsafe_allow_html=True)

            with right:
                st.markdown("**What to do**")
                st.markdown(f"""
<div class="section-card" style="border-left: 4px solid {color};">
<p style="color:#374151">{result['recommendation']}</p>
</div>
""", unsafe_allow_html=True)

            st.caption(
                "Logistic Regression · Threshold 0.35 · Recall-optimised · "
                "Trained on 7,043 Telco records · Treat probability as a risk ranking, not a guarantee."
            )


# BATCH SCORING TAB
# upload csv, score all, download results
with tab3:
    st.markdown("### Score a full customer list")
    st.markdown(
        "Upload a CSV of customer records. The model scores every customer and returns "
        "a prioritised list — highest risk first. Download the results when done."
    )

    st.info(
        "The CSV must include these columns: "
        "customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
        "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, "
        "TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, "
        "PaymentMethod, MonthlyCharges, TotalCharges."
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

    if uploaded:
        df_preview = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_preview):,} customers loaded** — first 5 rows:")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("Score All Customers", type="primary", use_container_width=True):
            uploaded.seek(0)
            file_bytes = uploaded.read()

            with st.spinner(f"Scoring {len(df_preview):,} customers..."):
                result = call_batch(file_bytes, uploaded.name)

            if result:
                preds = result["predictions"]

                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scored", f"{result['total_customers']:,}")
                c2.metric("High Risk", result["high_risk_count"])
                c3.metric("Medium Risk", result["medium_risk_count"])
                c4.metric("Revenue at Risk (High)", f"${result['total_revenue_at_risk']:,.0f}")

                # the main finding from the batch
                high_pct = round((result["high_risk_count"] / result["total_customers"]) * 100, 1)
                st.markdown(f"""
<div class="section-card" style="border-left: 5px solid #dc2626; background: #fef2f2;">
<b style="color:#991b1b">{high_pct}% of this batch are high risk.</b>
<span style="color:#374151"> Start from the top of the list below.
Revenue at risk covers the high-risk group only.</span>
</div>
""", unsafe_allow_html=True)

                # build download-ready dataframe
                df_out = pd.DataFrame([{
                    "Customer ID":        p["customerID"],
                    "Churn Probability":  f"{p['churn_probability']:.1%}",
                    "Risk Level":         p["risk_level"],
                    "Annual Revenue ($)": p["annual_revenue_at_risk"],
                    "Recommended Action": p["recommendation"],
                } for p in preds])

                st.markdown("### Prioritised customer list")
                st.caption("Sorted by churn probability — highest risk first.")
                st.dataframe(df_out, use_container_width=True)

                csv_bytes = df_out.to_csv(index=False).encode()
                st.download_button(
                    label="Download results as CSV",
                    data=csv_bytes,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# MODEL DETAILS TAB
# technical info for analysts — not the main story
with tab4:
    st.markdown("### Model details")
    st.caption("For analysts and technical reviewers who want to understand the model before acting on it.")

    info = get_model_info()

    if info:
        m1, m2, m3 = st.columns(3)
        m1.metric("Recall", f"{info['recall']:.1%}")
        m2.metric("ROC-AUC", info["roc_auc"])
        m3.metric("Decision Threshold", info["decision_threshold"])

        st.markdown("---")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Model setup**")
            st.markdown(f"""
<div class="section-card">
<p style="color:#374151; line-height:1.9">
Algorithm: <b>{info['model_type']}</b><br>
Class weighting: <b>{info['class_weighting']}</b><br>
Imbalance handling: <b>{info['imbalance_handling']}</b><br>
Priority metric: <b>{info['priority_metric']}</b><br>
Precision: <b>{info['precision']}</b><br>
Missed churners on test: <b>{info['missed_churners_on_test']} out of 374</b>
</p>
</div>
""", unsafe_allow_html=True)

            st.markdown("**Risk bands**")
            band_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#dc2626"}
            for band, rng in info["risk_bands"].items():
                color = band_colors.get(band, "#6b7280")
                st.markdown(
                    f"<span class='label-tag' style='background:{color}20; color:{color}; border:1px solid {color}40'>"
                    f"{band}</span> &nbsp; {rng}",
                    unsafe_allow_html=True
                )

        with right:
            st.markdown("**Top churn drivers**")
            for i, driver in enumerate(info["top_churn_drivers"], 1):
                st.markdown(f"**{i}.** {driver}")

            st.markdown("**Important note**")
            st.info(info["note"])

    else:
        st.warning("Could not load model info — make sure the API is running.")


# footer
st.divider()
st.markdown(
    "<div class='footer-note'>"
    "FastAPI &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; "
    "Logistic Regression &nbsp;·&nbsp; 7,043 records &nbsp;·&nbsp; "
    "<a href='https://godfreyadembesa.vercel.app' target='_blank'>godfreyadembesa.vercel.app</a>"
    "</div>",
    unsafe_allow_html=True
)