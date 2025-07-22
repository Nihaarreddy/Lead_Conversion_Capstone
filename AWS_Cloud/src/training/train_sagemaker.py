# src/training/train_sagemaker.py

import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing.cleaning import basic_cleaning
from src.preprocessing.pipelines import get_preprocessing_pipeline

# Constants
SEED = 42
TARGET_COL = "Converted"
MLFLOW_EXPERIMENT = "sagemaker_ml_experiment"
MLFLOW_MODEL_NAME = "Lead_Conversion_Prediction"
MLFLOW_MODEL_ARTIFACT_PATH = "best_pipeline"

# Basic model config (keep it simple for sagemaker training job)
MODEL = RandomForestClassifier(random_state=SEED)
MODEL_PARAMS = {"clf__n_estimators": [100, 200]}

def main(args):
    # Set MLflow Tracking URI (HOST THIS SEPARATELY OR POINT TO LOCAL DIR)
    mlflow.set_tracking_uri("file:///opt/ml/model/mlruns")  # or s3:// URI in production
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Read dataset
    input_file = os.path.join(args.input_data_path, "Lead Scoring.csv")
    df = pd.read_csv(input_file)
    df = basic_cleaning(df)

    # Split & Train
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)

    # Build Pipeline
    pipe = Pipeline([
        ("preprocessing", get_preprocessing_pipeline()),
        ("clf", MODEL)
    ])

    grid = GridSearchCV(pipe, MODEL_PARAMS, cv=3, scoring="roc_auc", verbose=2, n_jobs=-1)

    with mlflow.start_run():
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log everything
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "roc_auc": auc,
            "accuracy": acc,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path=MLFLOW_MODEL_ARTIFACT_PATH)

        # Save model to SageMaker expected path for hosting
        output_model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(grid.best_estimator_, output_model_path)

        print(f"✅ Model saved to: {output_model_path}")
        print(f"✅ AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker native input/output paths
    parser.add_argument('--input_data_path', type=str, default='/opt/ml/input/data/train')
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')

    args = parser.parse_args()
    main(args)
