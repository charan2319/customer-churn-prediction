"""
src/predict.py
--------------
Loads saved churn prediction pipeline and executes predictions on new customer raw input data.
"""

import os
import joblib
import pandas as pd


def load_pipeline(model_path: str = 'churn_prediction_pipeline.pkl'):
    """
    Loads serialized machine learning pipeline from disk.
    """
    if not os.path.exists(model_path):
        model_path = 'models/churn_pipeline.pkl'
    return joblib.load(model_path)


def predict_churn(customer_data: dict, model_path: str = 'churn_prediction_pipeline.pkl'):
    """
    Accepts raw customer data dictionary, applies saved pipeline, and returns prediction result.
    """
    pipeline = load_pipeline(model_path)

    # Convert single dictionary record into DataFrame
    df_input = pd.DataFrame([customer_data])

    # Generate prediction and probability
    pred_class = pipeline.predict(df_input)[0]
    prob_churn = pipeline.predict_proba(df_input)[0][1]

    churn_status = "Yes" if pred_class == 1 else "No"
    probability_pct = round(prob_churn * 100, 2)

    return {
        "Churn": churn_status,
        "Probability": f"{probability_pct}%",
        "Raw_Probability": prob_churn,
        "Class": pred_class
    }


if __name__ == '__main__':
    # Sample high-risk customer record
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.85,
        'TotalCharges': 179.70
    }

    result = predict_churn(sample_customer)
    print("--- PREDICTION RESULT FOR SAMPLE CUSTOMER ---")
    print(f"Prediction:        {result['Churn']}")
    print(f"Churn Probability: {result['Probability']}")
