# Credit Card Fraud Detection

This project is part of the GrowthLink Data Science Internship program. It focuses on building a machine learning pipeline to detect fraudulent credit card transactions with high accuracy and minimal false positives, using an imbalanced real-world dataset.



## Objective

The goal is to classify transactions as fraudulent or legitimate using machine learning techniques. We address data imbalance using SMOTE and implement hyperparameter tuning and advanced models to improve performance.



## Dataset

- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Description: 284,807 transactions, only 492 frauds
- Features: `V1` to `V28` (PCA components), `Time`, `Amount`
- Target: `Class` (0 = Legitimate, 1 = Fraud)


## Setup Instructions

1. Clone the repo:
   git clone https://github.com/Birundha2004/credit-card-fraud-detection
   cd credit-card-fraud-detection

2. Install dependencies:
   pip install -r requirements.txt

3. Open the Jupyter Notebook:
   jupyter notebook
   Run all cells in notebooks/FraudDetection.ipynb

4. Models Implemented
   Logistic Regression
   Random Forest Classifier (with GridSearchCV tuning)
   XGBoost Classifier (with ROC Curve)

5. Techniques Used
   SMOTE for class imbalance handling
   Feature scaling using StandardScaler
   Evaluation: Classification report, Confusion Matrix, ROC-AUC
   GridSearchCV for hyperparameter tuning

