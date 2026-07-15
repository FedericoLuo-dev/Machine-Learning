# Anonymized Loan Default Prediction

## Project Description
This project focuses on predicting loan defaults using a dataset with anonymized features. The workflow encompasses data preprocessing, exploratory data analysis (EDA), handling missing values with MCMC imputation (via `mice`), feature selection with `Boruta`, and building a series of classification models to predict the binary target variable `repay_fail`. The target indicates whether a loan was repaid ("No") or defaulted ("Si").

## Dataset
The dataset `Anonymize_Loan_Default_data.csv` includes various financial and applicant-specific features:
* **Selected Features for Modeling:** `int_rate`, `term`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `dti`, `purpose`, `total_acc`, `revol_bal`, `funded_amnt`, `total_pymnt`.
* **Target Variable:** `repay_fail` (mapped to "Si" for default and "No" for paid).

---

## Project Phases

### 1. Data Preprocessing & EDA
* Converted the target variable `repay_fail` into a binary factor ("Si" / "No").
* Visualized overall class distribution and missingness using the `VIM` package.
* **Missing Data Imputation:** Utilized the `mice` package with the CART (Classification and Regression Trees) method to impute missing values for the selected covariates.
* **Collinearity & Near-Zero Variance:** Checked for highly correlated predictors and near-zero variance features using `caret`. Handled a specific issue with `installment` and `funded_amnt`.

### 2. Feature Selection
* **Boruta:** Applied the Boruta algorithm to identify the most important features and confirmed the selection of the core variables.
* A baseline selection of variables was established for model training.

### 3. Model Training and Validation
Multiple classification models were trained and tuned using 10-fold Cross-Validation in `caret`. The evaluation metric optimized was Sensitivity ("Sens").
* **Logistic Regression (GLM)**
* **Naive Bayes**
* **Lasso Regression (GLMNET)**
* **K-Nearest Neighbors (KNN)**
* **Partial Least Squares (PLS)**
* **Decision Tree (CART/Rpart)**
* **Random Forest (RF)**
* **Gradient Boosting Machine (GBM)**
* **Neural Network (NNET)**

### 4. Model Evaluation & ROC Analysis
* Extracted predictions on the validation set.
* Generated ROC curves for all models using the `pROC` and `ROCR` packages.
* Overlaid ROC curves and calculated the Area Under the Curve (AUC) to compare models visually.
* Analyzed Lift and Gain charts using `funModeling` for the best performing models (e.g., GBM, GLM, NNET).

### 5. Threshold Optimization & Scoring
* Iterated over probability thresholds (0 to 1) to calculate Precision, Recall (Sensitivity), Specificity, and the F1-score for the Neural Network model predictions.
* Visualized the performance metrics against thresholds to find the optimal cut-off for classification.
* **Scoring:** Applied the chosen model (GBM) on the hold-out `score_set`, defining a custom threshold (0.05) to generate the final predictions and evaluated them using a Confusion Matrix.

![ROC curves](image_40327a.png)
