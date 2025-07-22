# 📁 train.py

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlflow.tracking import MlflowClient
from src.data_ingestion.db import read_redshift_table
from src.preprocessing.cleaning import basic_cleaning
from src.preprocessing.pipelines import get_preprocessing_pipeline
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# 🔧 Settings & Configurations
SEED = 42
TARGET_COL = "converted"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Define models and hyperparameters
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": RandomForestClassifier(random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    "decision_tree": DecisionTreeClassifier(random_state=SEED)
}

PARAMS = {
    "logistic_regression": {"clf__C": [1.0]},
    "random_forest": {"clf__n_estimators": [50, 100]},
    "gradient_boosting": {"clf__n_estimators": [50, 100]},
    "decision_tree": {"clf__max_depth": [3, 5, 10]}
}

# MLflow configuration (pointing to SageMaker Studio MLflow)
MLFLOW_TRACKING_URI = "arn:aws:sagemaker:ap-south-1:766957562594:mlflow-tracking-server/capstone-mlflow"
MLFLOW_EXPERIMENT = "leadconversionexperiment"
MLFLOW_MODEL_NAME = "LeadConversionPrediction"
MLFLOW_MODEL_ARTIFACT_PATH = "bestmodelpipeline"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# 📥 Load Data from S3 and clean it
def load_data():
    print("📥 Loading data from S3 using read_redshift_table()... (it now reads from S3)")
    df = read_redshift_table()  # Uses default path to S3 CSV from db.py
    print("🧹 Cleaning data...")
    df_clean = basic_cleaning(df)
    print(f"✅ Cleaned data shape: {df_clean.shape}")
    print("🧾 Columns in cleaned data:", df.columns.tolist())
    return df_clean

# 🧠 Training phase with MLflow tracking
def train_models(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=SEED)

    best_auc = -np.inf
    best_pipeline = None
    best_run_id = None

    with mlflow.start_run(run_name="final_train_pipeline"):
        for name, model in MODELS.items():
            print(f"🚀 Training model: {name}")
            pipeline = Pipeline([
                ("preprocessing", get_preprocessing_pipeline()),
                ("clf", model)
            ])
            search = GridSearchCV(pipeline, PARAMS[name], cv=3, scoring="roc_auc", verbose=1)

            with mlflow.start_run(run_name=name, nested=True):
                search.fit(X_train, y_train)
                y_pred = search.predict(X_test)
                y_proba = search.predict_proba(X_test)[:, 1]

                auc = roc_auc_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_params(search.best_params_)
                mlflow.log_param("model_name", name)
                mlflow.log_metrics({
                    "roc_auc": auc,
                    "f1_score": f1,
                    "accuracy": acc
                })

                mlflow.sklearn.log_model(search.best_estimator_, artifact_path=MLFLOW_MODEL_ARTIFACT_PATH)

                print(f"✅ {name} - AUC: {auc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_pipeline = search.best_estimator_
                    best_run_id = mlflow.active_run().info.run_id

    return best_pipeline, best_run_id

# 📦 Register Best Model to MLflow Registry
def register_pipeline(run_id):
    model_uri = f"runs:/{run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"
    client = MlflowClient()

    try:
        client.get_registered_model(MLFLOW_MODEL_NAME)
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"📦 Model '{MLFLOW_MODEL_NAME}' does not exist. Creating...")
            client.create_registered_model(MLFLOW_MODEL_NAME)
        else:
            raise

    # 👇 Register new version from the logged model
    model_ver = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )
    print(f"✅ Registered model '{MLFLOW_MODEL_NAME}' as version: {model_ver.version}")
    return model_ver

# 🚀 Entry point
if __name__ == "__main__":
    df = load_data()
    best_pipeline, run_id = train_models(df)
    register_pipeline(run_id)
    print("🏁 Training complete & model registered successfully!")
