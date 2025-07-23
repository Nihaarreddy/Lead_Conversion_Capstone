# Lead Conversion Prediction

Predicting Lead Conversion to Optimize Sales & Marketing

---

## 📈 Project Overview

This repository contains an end-to-end machine learning pipeline to **predict which leads are most likely to convert** into paying customers. The solution covers data ingestion, preprocessing, model building (Logistic Regression, Decision Tree, Random Forest), evaluation, deployment, and actionable business insights.

- **Business Problem:** Large portion of qualified leads fail to convert, resulting in lost resources and missed opportunities.
- **Goal:** Use robust, interpretable ML models on historic data to help sales focus on high-potential leads, improve ROI, and streamline lead management.

---

## 🌟 Key Features

- **Full pipeline**: Data extraction, cleaning, exploration, feature engineering, modeling, and visualization.
- **Config-driven workflow:** All preprocessing and pipeline steps are modular and easily configurable.
- **Robust metrics:** Accuracy, precision, recall, F1-score, confusion matrix, feature importances.
- **Deployment-ready:** ML models served via Flask web app for individual and batch scoring.
- **Experiment and drift tracking:** MLflow (for experiments), Evidently AI (for drift detection).
- **Business recommendations:** Actionable insights for marketing and sales teams.

---

## 📂 Repository Structure

<img width="959" height="792" alt="Screenshot 2025-07-22 102245" src="https://github.com/user-attachments/assets/ba183196-700f-4d3f-9db2-83bab1d08663" />

📦lead-conversion-prediction
┣ config/
┃ ┗ config.yaml
┣ notebooks/
┃ ┗ eda.ipynb
┣ src/
┃ ┣ preprocessing
        cleaning.py
        feature_engineering.py
        pipelines.py
┃ ┣ model
        train.py
┃ ┣ monitoring
        drifts.py
        generate_drift_reports.py
┃ ┗ api/
        templates/
         app.py
┃ ┣ flask_app.py
┃ ┗ data_ingestion
        db.py
┃ ┗ explainability
        shap_explainer.py
┣ mlruns/
┣ reports/
┃ ┣ drift_full_vs_batch.html
┃ ┗ drift_train_test.html
┣ output-screenshots/
┃ ┣ Drift_Execution.png
┃ ┣ MLFlow UI.png
┃ ┗ Models_compare_Visualize.png
┣ requirements.txt
┣ README.md
┗ LICENSE



---

## 🚀 Quick Start

### 1. Clone the repository

git clone https://github.com/Nihaarreddy/Lead_Conversion_Capstone.git
cd Lead_Conversion_Capstone


### 2. Install dependencies


pip install -r requirements.txt


### 3. Configure and prepare your environment

- Setup environment variables as needed (DB URIs, API keys).
- Update `config/config.yaml` for feature groups or pipeline changes.

### 4. Run Data Preparation & Modeling

Use the included Jupyter notebooks or scripts in `src/`:

python src/preprocessing/data_cleaning.py
python src/preprocessing/pipelines.py



Or use the main pipeline script:

python src/pipeline.py --train

### 5. Serve Models via Flask UI

python src/app/flask_app.py


- Access UI at `http://localhost:8000 

<img width="1919" height="1005" alt="Screenshot 2025-07-21 182850" src="https://github.com/user-attachments/assets/56cbf141-ed7c-47b1-8d73-5d45e5df1cc0" />


<img width="1917" height="949" alt="Screenshot 2025-07-21 182903" src="https://github.com/user-attachments/assets/53f5f329-1c4a-4fc9-9aa9-3d83aa83592c" />

---

## 📊 Model Performance

| Model              | Precision | Recall | F1-score | Not Converted | Converted |
|---------------------|-----------|--------|----------|---------------|-----------|
| **Decision Tree**   | 0.92-0.95 | 0.98-0.83 | 0.93 | 3365 | 1594 |
| **Random Forest**   | 0.99      | 0.99   | 0.99    | 4401          | 2055      |
| **Logistic Regression** | 0.98-0.97 | 0.99-0.95 | 0.97 | 4432 | 2024 |

> For detailed confusion matrices and feature importances, see `reports/` and visualizations in `notebooks/`.

---

## 🔍 Feature Engineering & Pipeline

- **Numeric:** Median imputation, MinMax scaling
- **Categorical:** One-hot encoding after rare-category grouping
- **Binary:** Yes/No mapped to 1/0
- **Engineered Features:**
    - `VisitsPerPage = TotalVisits / (Page Views Per Visit + 1)`
    - `ActivityScoreRatio = ProfileScore / ActivityScore`
    - Missingness indicators and rare-group bins
- **Pipeline Example:**
    - Cleaning → Feature Eng → Rare Grouping → Encoding/Scaling → Variance Filter → Modeling

---

## 🛠️ Deployment & Monitoring

- **MLflow:** All runs and models logged/tracked for reproducibility
- **Evidently AI:** Regular drift detection with HTML reports, see sample:  

<img width="1918" height="959" alt="Screenshot 2025-07-21 182932" src="https://github.com/user-attachments/assets/61a561b4-88ed-4199-941d-0fe498522925" />


- **Flask UI:**  
    - **Single Lead Scoring:** Real-time form input
    - **Batch Scoring:** CSV file upload for bulk predictions

---

## 📢 Business Insights

- **Top Predictive Features:** Engagement activity, lead source, explicit missingness
- **Recommendations:**
  - Prioritize high-title-score, highly engaged leads for sales
  - Optimize marketing spend on lead sources with high predicted conversion
  - Monitor/clean "Other" fields that may hide poor or strong segments

---

## 📝 Appendices

- **Example Code:** See `notebooks/` for step-by-step reproducible analysis
- **Data Dictionary & Glossary:** Provided in `/docs/`
- **Keep Improving:** Contributions, issues, and feature requests are welcome!

---

## 📸 Screenshots

<!-- Add screenshots here -->
<img width="1914" height="950" alt="All_Models_Params_Metrics" src="https://github.com/user-attachments/assets/eaab4eab-8f4b-4f03-8d6d-9b5e43660f8c" />


<img width="1882" height="594" alt="MLFlow-Registered-Models" src="https://github.com/user-attachments/assets/6f78700f-6e3d-41b4-b3a4-7440267eb730" />

---

## 🤝 Contributions

Open to suggestions, improvements, bug fixes, and collaboration! Please open an issue or submit a pull request.

---

