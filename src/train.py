"""
src/train.py
------------
Trains the complete end-to-end Machine Learning Pipeline and saves the fitted model object.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from preprocessing import load_and_clean_data, prepare_features_and_target, split_data, get_preprocessor


def train_and_save_pipeline(data_path: str = 'data/Telco-Customer-Churn.csv', model_output_path: str = 'models/churn_pipeline.pkl'):
    """
    Trains the full end-to-end machine learning pipeline and serializes it with joblib.
    """
    print("1. Loading raw dataset...")
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df)

    print("2. Splitting dataset (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("3. Building end-to-end Pipeline...")
    preprocessor = get_preprocessor()

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
        ('classifier', LogisticRegression(C=10, max_iter=1000, random_state=42))
    ])

    print("4. Fitting complete pipeline on X_train...")
    final_pipeline.fit(X_train, y_train)

    print("5. Evaluating pipeline on untouched X_test...")
    y_pred = final_pipeline.predict(X_test)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]

    print("\n--- TEST METRICS ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    print(f"\n6. Saving fitted pipeline to '{model_output_path}'...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_pipeline, model_output_path)
    print("Pipeline successfully saved!")
    return final_pipeline


if __name__ == '__main__':
    train_and_save_pipeline()
