"""
src/preprocessing.py
-------------------
Data cleaning, feature preparation, train/test split, and Scikit-Learn Preprocessing Pipelines.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Feature lists
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def load_and_clean_data(csv_path: str):
    """
    Loads raw CSV data, converts TotalCharges to numeric (coercing whitespace " " to NaN),
    and returns cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Separates feature matrix X and target vector y.
    Removes customerID identifier column.
    Maps target variable: No -> 0, Yes -> 1.
    """
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'No': 0, 'Yes': 1})
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """
    Splits dataset into stratified train and test sets (80% train / 20% test).
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_preprocessor():
    """
    Creates and returns the ColumnTransformer preprocessing pipeline.
    - Numerical: SimpleImputer(median) -> StandardScaler()
    - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, NUMERICAL_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


if __name__ == '__main__':
    data_path = 'data/Telco-Customer-Churn.csv'
    df = load_and_clean_data(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = get_preprocessor()
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Phase 4 Preprocessing Pipeline Execution Successful!")
    print(f"Original Feature Count: {X_train.shape[1]}")
    print(f"Processed Feature Count: {X_train_processed.shape[1]}")
    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape:  {X_test_processed.shape}")
