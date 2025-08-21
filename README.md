# Telco Customer Churn Analysis

## Overview
This project predicts customer churn for a telecommunications company using the Telco Customer Churn dataset (7,043 records, 21 features). Key features include `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, and the binary target `Churn` (1=Yes, 0=No). The project involves data preprocessing, visualization, and modeling with Logistic Regression and Random Forest.

## Project Steps
- **Preprocessing**:
  - Dropped `customerID` (irrelevant).
  - Converted `TotalCharges` to numeric, dropped missing values and duplicates.
  - Merged "No phone/internet service" into "No" for columns like `MultipleLines` and `OnlineSecurity`.
  - Created dummy variables (`drop_first=True`) to avoid the dummy variable trap.
  - Scaled features to [0,1] using MinMaxScaler.
  - Split data: 75% training, 25% testing.

- **Visualization**:
  - Correlation heatmap: `tenure` (-0.354) and `Contract_Two year` (-0.302) reduce churn; `InternetService_Fiber optic` (0.307) and `PaymentMethod_Electronic check` (0.301) increase it.
  - Tenure histogram: Most customers at 0-1 or ~72 months, indicating high early churn and long-term retention.
  - MonthlyCharges vs. TotalCharges scatter: Strong positive linear relationship, suggesting higher charges link to churn.
  - Tenure box plot: Non-churners have longer tenure (median ~38 months) vs. churners (~10 months).

- **Modeling**:
  - **Logistic Regression**: Accuracy (0.792), Precision (0.619), Recall (0.517). Better at identifying churners.
  - **Random Forest**: Accuracy (0.795), OOB Error (0.195), Precision (0.656), Recall (0.450). Higher precision but misses more churners.

- **Evaluation**:
  - Confusion matrices: Logistic Regression (1154 TN, 237 TP, 146 FP, 221 FN) balances churn detection; Random Forest (1192 TN, 206 TP, 108 FP, 252 FN) prioritizes non-churn accuracy.
  - Logistic Regression is more suitable due to higher recall, critical for targeting churners in retention strategies.

## Files
- `telco_customer_churn.ipynb`: Jupyter notebook with all code.
- `Telco-Customer-Churn.csv`: Dataset.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `scikit-learn`, `matplotlib`

## Setup
1. Clone the repo: [HERE](https://github.com/XolaniGatebe/Telco-Customer-Churn-Analysis.git).
2. Install dependencies: `pip install -r requirements.txt`
3. Run `telco_customer_churn.ipynb` in Jupyter Notebook.

## Author
Xolani Gatebe.
