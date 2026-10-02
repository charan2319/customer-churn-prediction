# Customer Churn Prediction — End-to-End Machine Learning Pipeline

An end-to-end binary classification machine learning system that predicts whether a telecommunications customer is likely to churn based on customer demographic details, service usage, contract terms, and billing information.

---

## 📌 Problem Statement

Customer attrition (churn) is a critical challenge in the telecom industry. Acquiring a new customer is significantly more expensive than retaining an existing subscriber. The goal of this project is to build an interpretable, reproducible machine learning pipeline that predicts customer churn propensity (`No = 0`, `Yes = 1`), enabling business teams to execute proactive retention strategies.

---

## 🎯 Objective

1. **Understand & Clean Data:** Identify hidden data-quality issues (such as blank whitespace strings in `TotalCharges`) and prepare the dataset for statistical modeling.
2. **Prevent Data Leakage:** Build an end-to-end Scikit-Learn `Pipeline` that encapsulates missing-value imputation, feature scaling, one-hot encoding, feature selection, and classification.
3. **Optimize Model Performance:** Perform 5-fold Stratified Cross-Validation and hyperparameter tuning using `GridSearchCV`.
4. **Deploy Web Application:** Serve the saved model pipeline using **Streamlit** to enable real-time inference on raw 19-feature customer profiles.

---

## 📊 Dataset Overview

* **Source:** IBM Telco Customer Churn Dataset (`data/Telco-Customer-Churn.csv`)
* **Total Instances:** `7,043` rows
* **Total Features:** `21` columns (1 target, 1 identifier, 3 numerical, 16 categorical)
* **Target Variable (`Churn`):** Binary classification
  * `No` (Retained): **5,174** (73.46%)
  * `Yes` (Churned): **1,869** (26.54%)

---

## 🔄 Machine Learning Workflow Architecture

```text
               IBM Telco Customer Churn Dataset
                              ↓
                     Data Understanding
                              ↓
                 Exploratory Data Analysis (`src/eda.py`)
                              ↓
                        Data Cleaning
                     (TotalCharges → float)
                              ↓
                      Train/Test Split
                     (80% Train / 20% Test)
                              ↓
                    Scikit-Learn Pipeline
  ┌───────────────────────────┴───────────────────────────┐
Numerical Features                                Categorical Features
SimpleImputer(strategy="median")                  SimpleImputer(strategy="most_frequent")
StandardScaler()                                  OneHotEncoder(handle_unknown="ignore")
  └───────────────────────────┬───────────────────────────┘
                              ↓
                   (46 Preprocessed Features)
                              ↓
                      SelectKBest (k=20)
                     (ANOVA f_classif Test)
                              ↓
                    Logistic Regression
                      (C=10, max_iter=1000)
                              ↓
                Hyperparameter Tuning (GridSearchCV)
                              ↓
                     Final Test Evaluation
                              ↓
                 Streamlit Web Application
```

---

## 🛠️ Data Preprocessing & Leakage Prevention

* **Train/Test Split:** Applied `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` prior to fitting any transformers to ensure zero data leakage from test data to train data.
* **Numerical Features (`tenure`, `MonthlyCharges`, `TotalCharges`):** Imputed missing values using `SimpleImputer(strategy="median")` and standardized to zero mean ($\mu=0$) and unit variance ($\sigma=1$) using `StandardScaler()`.
* **Categorical Features (16 variables):** Imputed using `SimpleImputer(strategy="most_frequent")` and encoded into binary columns using `OneHotEncoder(handle_unknown="ignore")`.
* **Feature Expansion:** Expanded the 19 input features into **46 preprocessed features**.

---

## 🔍 Feature Selection (`SelectKBest`)

* **Algorithm:** `SelectKBest(score_func=f_classif, k=20)`
* **Methodology:** Conducts an ANOVA F-test between each preprocessed feature and the target label (`Churn`). Features with higher F-scores indicate stronger statistical association with churn.
* **Dimension Reduction:** Reduced feature space from **46 down to the 20 most statistically associated features**.
* **Top 5 Features by F-Score:**
  1. `Contract_Month-to-month` (F-Score: 1,114.22)
  2. `OnlineSecurity_No` (F-Score: 767.86)
  3. `tenure` (F-Score: 763.89)
  4. `TechSupport_No` (F-Score: 731.77)
  5. `InternetService_Fiber optic` (F-Score: 610.20)

---

## 🤖 Machine Learning Model & Hyperparameter Tuning

* **Classifier:** `LogisticRegression(C=10, max_iter=1000, random_state=42)`
* **Why Logistic Regression?**
  * Naturally suited for binary classification problems.
  * Simple, interpretable, and computationally lightweight.
  * Outputs calibrated probabilities (`predict_proba`) required for risk scoring.
  * Operates effectively on standardized numerical and one-hot encoded binary features.
* **Hyperparameter Tuning:** Evaluated $C \in [0.01, 0.1, 1, 10, 100]$ using 5-fold Stratified Cross-Validation (`GridSearchCV`). Optimal performance achieved at **$C = 10$**.

---

## 📈 Model Performance (Untouched Test Set Evaluation)

Evaluating the final pipeline on the 20% test set ($n = 1,409$):

| Metric | Test Set Result | Description |
| :--- | :--- | :--- |
| **Accuracy** | **79.49%** | Overall percentage of correctly classified customers. |
| **Precision** | **63.49%** | Percentage of true churners among all predicted churners. |
| **Recall** | **53.48%** | Percentage of actual churners correctly flagged by model. |
| **F1-Score** | **0.5806** | Harmonic mean of Precision and Recall on minority class. |
| **ROC-AUC** | **0.8392** | Ability of model to discriminate churners across thresholds. |

---

## 💻 Streamlit Web Application

The application (`app.py`) loads the complete serialized pipeline (`churn_prediction_pipeline.pkl`) and accepts raw customer profile inputs.

### Features & Workflow:
1. **Raw Input Acceptance:** Users enter 19 raw customer attributes via clean Streamlit selection controls.
2. **Direct Pipeline Inference:** Raw DataFrame is passed directly to `pipeline.predict(df_input)` without manual preprocessing code.
3. **Probability Display:** Outputs Churn Probability, Retention Probability, Metric Cards, and visual Progress Bars.

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── app.py                         # Streamlit Web Application
├── churn_prediction_pipeline.pkl  # Complete Saved Scikit-Learn Pipeline (Deployment)
├── requirements.txt               # Minimal Project Dependencies
├── README.md                      # Comprehensive Project Documentation
├── .gitignore                     # Git Exclusions File
│
├── data/
│   └── Telco-Customer-Churn.csv   # Raw Customer Dataset
│
├── models/
│   └── churn_pipeline.pkl         # Backup Model Pipeline Copy
│
└── src/
    ├── eda.py                     # Programmatic Exploratory Data Analysis Script
    ├── preprocessing.py           # Data Loading, Cleaning & Preprocessor Pipeline
    ├── train.py                   # Model Training & Pipeline Serialization Script
    └── predict.py                 # Standalone Inference Script
```

---

## 🚀 Local Installation & Execution

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/charan2319/customer-churn-prediction.git
cd customer-churn-prediction

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run EDA Analysis
```bash
python src/eda.py
```

### 3. Run Streamlit Application
```bash
streamlit run app.py
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push code repository to GitHub.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app** $\rightarrow$ Select repository `customer-churn-prediction` $\rightarrow$ Branch `main` $\rightarrow$ Main file `app.py`.
4. Click **Deploy!**

---

## 🔮 Future Improvements

1. **Threshold Tuning:** Adjust decision threshold below 0.5 to increase Recall for high-value customer retention campaigns.
2. **SMOTE Resampling:** Explore Synthetic Minority Over-sampling Technique inside pipeline to address class imbalance.
3. **Feature Engineering:** Create custom interaction ratios (e.g., `TotalCharges / tenure`).
