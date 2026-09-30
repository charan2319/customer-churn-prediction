"""
app.py
------
Streamlit Web Application for Customer Churn Prediction.
Loads the complete saved Scikit-Learn pipeline and generates real-time predictions.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Single primary model file path for deployment
MODEL_PATH = "churn_prediction_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/churn_pipeline.pkl"


@st.cache_resource
def load_churn_pipeline():
    """Loads and caches the serialized scikit-learn pipeline."""
    return joblib.load(MODEL_PATH)


# Header & Title
st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict the likelihood of a telecom customer churning based on demographic details, "
    "account information, and subscribed services."
)
st.divider()

# Load Model Pipeline with User-Friendly Error Handling
try:
    pipeline = load_churn_pipeline()
except Exception:
    st.error("⚠️ Unable to load the trained model pipeline file (`churn_prediction_pipeline.pkl`). Please ensure the model file is present.")
    st.stop()

# Input Form Layout
st.header("👤 Customer Profile Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Customer Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_str == "Yes" else 0
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    st.subheader("2. Account Details")
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, value=12, step=1,
        help="Number of months the customer has stayed with the company."
    )
    contract = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5
    )
    total_charges_input = st.text_input(
        "Total Charges ($) (Leave empty for new customers)",
        value="780.0",
        help="Leave blank or enter float value."
    )
    # Process TotalCharges input safely
    try:
        total_charges = float(total_charges_input) if total_charges_input.strip() != "" else np.nan
    except ValueError:
        total_charges = np.nan

with col3:
    st.subheader("3. Billing & Payment")
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

st.divider()

st.subheader("4. Subscribed Services")
scol1, scol2, scol3 = st.columns(3)

with scol1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "Internet Service", ["Fiber optic", "DSL", "No"]
    )

with scol2:
    online_security = st.selectbox(
        "Online Security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection", ["No", "Yes", "No internet service"]
    )

with scol3:
    tech_support = st.selectbox(
        "Tech Support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", ["No", "Yes", "No internet service"]
    )

st.divider()

# Prediction Action
if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
    # Build raw input DataFrame (19 features)
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
        # Generate prediction and probability using saved pipeline directly
        prediction = pipeline.predict(raw_input_df)[0]
        probabilities = pipeline.predict_proba(raw_input_df)[0]

        churn_prob = float(probabilities[1])
        retention_prob = float(probabilities[0])

        st.subheader("🎯 Prediction Output")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.error("⚠️ **Churn Prediction: Customer is likely to CHURN.**")
            else:
                st.success("✅ **Churn Prediction: Customer is likely to STAY (Retained).**")

            st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
            st.metric("Retention Probability", f"{retention_prob * 100:.1f}%")

        with res_col2:
            st.write("**Probability Distribution**")
            st.write(f"Churn Risk: **{churn_prob * 100:.1f}%**")
            st.progress(churn_prob)
            st.write(f"Retention Likelihood: **{retention_prob * 100:.1f}%**")
            st.progress(retention_prob)

    except Exception:
        st.error("Unable to generate prediction. Please verify the entered inputs.")
