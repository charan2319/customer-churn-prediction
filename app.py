"""
app.py
------
Customer Churn Prediction Enterprise Dashboard.
Loads the serialized Scikit-Learn pipeline and executes real-time inference.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Telecom Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for human enterprise styling
st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        color: #991B1B;
        border-radius: 6px;
        padding: 1rem;
        font-weight: 600;
    }
    .risk-low {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        color: #166534;
        border-radius: 6px;
        padding: 1rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Primary model file path
MODEL_PATH = "churn_prediction_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/churn_pipeline.pkl"


@st.cache_resource
def load_churn_pipeline():
    """Loads and caches the serialized scikit-learn pipeline."""
    return joblib.load(MODEL_PATH)


# Sidebar Metadata & Information
with st.sidebar:
    st.title("System Metadata")
    st.markdown("**Model Pipeline:** End-to-End Scikit-Learn")
    st.markdown("**Classifier:** Logistic Regression ($C=10$)")
    st.markdown("**Feature Selection:** SelectKBest ($k=20$, ANOVA F-test)")
    st.markdown("**Test Set ROC-AUC:** `0.8392`")
    st.markdown("**Test Set Accuracy:** `79.49%`")
    st.divider()
    st.markdown("**Dataset Source:** IBM Telco Customer Churn")
    st.markdown("**Pipeline Serialization:** Joblib (`churn_prediction_pipeline.pkl`)")

# Header Section
st.markdown('<div class="main-header">Customer Retention & Churn Scoring Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Input customer demographic, account, and service parameters to compute real-time churn risk.</div>', unsafe_allow_html=True)

# Load Model Pipeline
try:
    pipeline = load_churn_pipeline()
except Exception:
    st.error("Unable to load the model pipeline file. Please ensure `churn_prediction_pipeline.pkl` is present in the working directory.")
    st.stop()

# Input Form in Structured Tabs
tab_account, tab_services = st.tabs(["Account & Demographics", "Service Subscriptions"])

with tab_account:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographic Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
        senior_citizen = 1 if senior_citizen_str == "Yes" else 0
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    with col2:
        st.subheader("Account & Billing Details")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, step=1)
        contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total_charges_input = st.text_input("Total Charges ($) [Leave blank if unknown]", value="780.0")
        
        try:
            total_charges = float(total_charges_input) if total_charges_input.strip() != "" else np.nan
        except ValueError:
            total_charges = np.nan

        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

with tab_services:
    scol1, scol2 = st.columns(2)

    with scol1:
        st.subheader("Connectivity Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])

    with scol2:
        st.subheader("Security & Support Add-ons")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Trigger Button
if st.button("Evaluate Churn Risk", type="primary", use_container_width=True):
    raw_input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = pipeline.predict(raw_input_df)[0]
        probabilities = pipeline.predict_proba(raw_input_df)[0]

        churn_prob = float(probabilities[1])
        retention_prob = float(probabilities[0])

        st.markdown("### Risk Evaluation Summary")

        rcol1, rcol2, rcol3 = st.columns([2, 1, 1])

        with rcol1:
            if prediction == 1:
                st.markdown(
                    f'<div class="risk-high">High Churn Propensity Detected<br>'
                    f'<span style="font-weight:400; font-size:0.9rem;">Customer exhibits high risk of cancellation within the next billing cycle.</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="risk-low">Customer Retained (Low Churn Risk)<br>'
                    f'<span style="font-weight:400; font-size:0.9rem;">Customer profile indicates strong retention probability.</span></div>',
                    unsafe_allow_html=True
                )

        with rcol2:
            st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")

        with rcol3:
            st.metric("Retention Probability", f"{retention_prob * 100:.1f}%")

        # Key Risk Factors Narrative
        st.markdown("#### Key Profile Observations")
        observations = []
        if contract == "Month-to-month":
            observations.append("• Month-to-month contract is associated with higher historical customer attrition.")
        if internet_service == "Fiber optic":
            observations.append("• Fiber optic subscription represents a higher monthly expenditure bracket.")
        if tenure <= 12:
            observations.append("• Short tenure (<= 12 months) indicates early customer lifecycle vulnerability.")
        if payment_method == "Electronic check":
            observations.append("• Manual electronic check payments show higher observed attrition than automated payments.")
        if online_security == "No" or tech_support == "No":
            observations.append("• Absence of value-added security/support add-ons is correlated with higher churn rates.")

        if not observations:
            observations.append("• Customer exhibits strong retention factors (long tenure, multi-year contract, automated billing).")

        for obs in observations:
            st.write(obs)

    except Exception as err:
        st.error(f"Error executing model pipeline: {err}")
