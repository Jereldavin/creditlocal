# 💳 Credit Score Classification - Monolithic Streamlit Cloud App & ML Pipeline
This repository contains an end-to-end Machine Learning pipeline and an interactive web application for classifying customer credit score tiers (**Good**, **Standard**, and **Bad**). The application is deployed live on **Streamlit Community Cloud** using a monolithic architecture.
🌐 **Live Streamlit Demo:** [https://creditlocal-bq7n6692srwyee6d9mkzxg.streamlit.app/](https://creditlocal-bq7n6692srwyee6d9mkzxg.streamlit.app/)
---
## 📌 Project Overview
- **Exploratory Data Analysis (EDA):** Performed data quality audit on `data_A.csv`, cleaned contaminated numeric columns via regular expressions, handled unrealistic outliers using masking, and inspected categorical feature distributions.
- **Object-Oriented Programming (OOP) Pipeline:** Engineered modular classes adhering to single-responsibility design principles:
  - `DataIngestion`: Verifies source files, validates non-empty payloads, and creates destination ingestion paths.
  - `DataPreprocessor`: Strips corrupted string characters, enforces numerical casting, handles outliers, and removes Personally Identifiable Information (PII).
  - `ModelTrainer`: Prepares scikit-learn `ColumnTransformer` (median imputation for skewed features, mean for symmetric, and one-hot encoding for categorical features), trains models (`Logistic Regression`, `Random Forest`, `Gradient Boosting`), and logs experiments to MLflow.
  - `ModelEvaluator`: Evaluates holdout test predictions, logs accuracy metrics to MLflow runs, identifies the winning estimator, and exports the serialized model.
  - `MLEndToEndPipeline`: Orchestrates execution flow from raw input ingestion to artifact export.
- **Monolithic Cloud Deployment:** Hosted directly on Streamlit Community Cloud, bundling the trained model artifact (`best_credit_model.pkl`) to deliver sub-second, in-memory inference without external API overhead.
