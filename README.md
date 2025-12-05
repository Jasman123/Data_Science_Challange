# 🌟 Loan Default Prediction — Machine Learning Pipeline

### *End-to-End Credit Risk Modeling for Data Science Challenge*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML%20Models-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/XGBoost-Boosting-orange" />
  <img src="https://img.shields.io/badge/LightGBM-Gradient%20Boosting-11AA6C" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/EDA-Exploratory%20Analysis-purple" />
</p>

---

# 📘 Overview

This repository contains a complete **end-to-end Machine Learning pipeline** for predicting **loan default risk**.

It includes:

* Exploratory Data Analysis (EDA)
* Feature engineering & preprocessing
* Model development (XGBoost & LightGBM)
* Evaluation & interpretability
* Final prediction generation for submission

The core implementation lives in:

📌 **LoanDefaultPrediction.ipynb**

---

# ⚙️ Features

### 🔍 Full Exploratory Data Analysis
Distribution plots, correlation heatmaps, missing value checks & insights.

### 🧼 Automated Preprocessing Pipeline
Handles:
* Missing values  
* Categorical encoding  
* Numerical scaling  
* Outlier handling  

### 🧠 Advanced ML Models
Two high-performing boosting models:
* **XGBoost**
* **LightGBM**

### 📈 Feature Importance & Explainability
Gain-based importance + optional SHAP analysis.

### 📊 Submission-Ready Output
Automatically generates:

---
prediction_submission.csv


---

# 🧠 How It Works

The workflow follows this ML pipeline:

1. Load training & test datasets  
2. Perform EDA  
3. Clean & preprocess data  
4. Engineer new features  
5. Train XGBoost & LightGBM models  
6. Evaluate models  
7. Save best models (`.pkl`)  
8. Generate predictions for the test file  

This ensures a **clean, reproducible** machine learning process.

---

# 🧩 Code Breakdown

## 1. Data Loading

```python
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
```
