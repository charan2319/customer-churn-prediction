"""
app.py
------
Streamlit Web Application for Customer Churn Prediction.
Loads the complete saved Scikit-Learn pipeline and generates real-time predictions.
"""

import os
import base64
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# Hide Streamlit top-right toolbar, share/edit/github buttons, header, and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    .stAppHeader {visibility: hidden !important; display: none !important;}
    /* Hide header title anchor link chain icons */
    .header-anchor {display: none !important;}
    [data-testid="stHeaderActionElements"] {display: none !important;}
    a.anchor-link {display: none !important;}
    /* Reduce top whitespace above title */
    .block-container, [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to convert image to base64 for inline flexbox layout
def get_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Model file path
MODEL_PATH = "models/churn_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "churn_prediction_pipeline.pkl"


@st.cache_resource
def load_churn_pipeline():
    """Loads and caches the serialized scikit-learn pipeline."""
    return joblib.load(MODEL_PATH)


# Load Model Pipeline
try:
    pipeline = load_churn_pipeline()
except Exception:
    st.error("Unable to load the model pipeline file (`churn_prediction_pipeline.pkl`). Please ensure the model file is present.")
    st.stop()

# Header Section
st.title("📊 Customer Churn Prediction")
st.write("Input customer demographics, account details, and subscribed services to predict churn probability.")
st.markdown("---")

# Top Section: Demographics & Contract Billing (2 Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics & Account")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_str == "Yes" else 0
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, step=1)

with col2:
    st.subheader("📄 Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges_input = st.text_input("Total Charges ($) [Optional]", value="780.0")
    
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

st.markdown("---")

# Bottom Section: Subscribed Services (Spread across 3 Columns)
st.subheader("🛠️ Subscribed Services")
scol1, scol2, scol3 = st.columns(3)

with scol1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

with scol2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with scol3:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.markdown("---")

# Prediction Trigger
if st.button("Predict Churn", type="primary", use_container_width=True):
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

        st.subheader("🎯 Prediction Result")

        rcol1, rcol2, rcol3 = st.columns(3)

        with rcol1:
            if prediction == 1:
                if os.path.exists("assets/churn_warning.png"):
                    b64_img = get_image_b64("assets/churn_warning.png")
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px;">
                            <img src="data:image/png;base64,{b64_img}" width="50" style="object-fit: contain;" />
                            <span style="color: #DC2626; font-size: 1.1rem; font-weight: 700;">Customer is likely to CHURN</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: #DC2626; font-size: 1.1rem; font-weight: 700;">Customer is likely to CHURN</span>', unsafe_allow_html=True)
            else:
                if os.path.exists("assets/customer_retained.png"):
                    b64_img = get_image_b64("assets/customer_retained.png")
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px;">
                            <img src="data:image/png;base64,{b64_img}" width="50" style="object-fit: contain;" />
                            <span style="color: #16A34A; font-size: 1.1rem; font-weight: 700;">Customer is likely to STAY (Retained)</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<span style="color: #16A34A; font-size: 1.1rem; font-weight: 700;">Customer is likely to STAY (Retained)</span>', unsafe_allow_html=True)

        with rcol2:
            st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")

        with rcol3:
            st.metric("Retention Probability", f"{retention_prob * 100:.1f}%")

    except Exception:
        st.error("Unable to generate prediction. Please check the entered values.")
