"""
src/eda.py
----------
Exploratory Data Analysis (EDA) script for the Telco Customer Churn dataset.
Calculates statistical distributions, class imbalances, and feature associations.
"""

import pandas as pd


def run_eda(csv_path: str = 'data/Telco-Customer-Churn.csv'):
    """
    Executes exploratory data analysis on the raw dataset and prints summary findings.
    """
    print("=" * 60)
    print("      EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(csv_path)
    df['TotalCharges_num'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 1. Dataset Shape & Missing Values
    print("\n1. DATASET OVERVIEW")
    print(f"   Total Instances: {df.shape[0]} rows")
    print(f"   Total Attributes: {df.shape[1]} columns")
    print(f"   TotalCharges Blank Spaces (' '): {df['TotalCharges'].astype(str).str.strip().eq('').sum()} rows")

    # 2. Target Variable Distribution
    print("\n2. TARGET DISTRIBUTION (Churn)")
    target_counts = df['Churn'].value_counts()
    target_pct = df['Churn'].value_counts(normalize=True) * 100
    for category in target_counts.index:
        count = target_counts[category]
        pct = target_pct[category]
        print(f"   {category:<5}: {count:5d} ({pct:.2f}%)")

    # 3. Contract Type vs Observed Churn Rate
    print("\n3. OBSERVED CHURN RATE BY CONTRACT TYPE")
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for contract, rate in contract_churn.items():
        print(f"   {contract:<16}: {rate:5.2f}% Churn Rate")

    # 4. Tenure Statistics by Churn Status
    print("\n4. TENURE STATISTICS BY CHURN STATUS (Months)")
    tenure_stats = df.groupby('Churn')['tenure'].describe()[['min', '25%', '50%', '75%', 'max', 'mean']]
    print(tenure_stats.round(2).to_string())

    # 5. Monthly Charges Statistics by Churn Status
    print("\n5. MONTHLY CHARGES STATISTICS BY CHURN STATUS ($)")
    monthly_stats = df.groupby('Churn')['MonthlyCharges'].describe()[['min', '25%', '50%', '75%', 'max', 'mean']]
    print(monthly_stats.round(2).to_string())

    # 6. Internet Service vs Churn Rate
    print("\n6. OBSERVED CHURN RATE BY INTERNET SERVICE")
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    for service, rate in internet_churn.items():
        print(f"   {service:<16}: {rate:5.2f}% Churn Rate")

    # 7. Payment Method vs Churn Rate
    print("\n7. OBSERVED CHURN RATE BY PAYMENT METHOD")
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    for method, rate in payment_churn.items():
        print(f"   {method:<26}: {rate:5.2f}% Churn Rate")

    # 8. Numerical Feature Correlation Matrix
    print("\n8. PEARSON CORRELATION MATRIX (Numerical Features & Binary Churn)")
    df_num = df[['tenure', 'MonthlyCharges', 'TotalCharges_num']].copy()
    df_num['Churn'] = (df['Churn'] == 'Yes').astype(int)
    print(df_num.corr().round(3).to_string())

    print("\n" + "=" * 60)
    print("      END OF EDA REPORT")
    print("=" * 60)


if __name__ == '__main__':
    run_eda()
