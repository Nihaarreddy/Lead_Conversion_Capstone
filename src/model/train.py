"""
train.py
Complete training with three phases:

Phase 1:
    Prototype on 20% sample for quick experimentation.
Phase 2:
    Full data 80/20 split for honest model selection.
Phase 3:
    Retrain best model on 100% data for production and register in MLflow Model Registry.

Each phase logs artifacts to MLflow and saves the best pipeline for deployment.
"""
# Importing all the Libraries required
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.preprocessing.cleaning import basic_cleaning
from src.preprocessing.pipelines import get_preprocessing_pipeline
from src.data_ingestion.db import read_table
from src.explainability.shap_explainer import explain_model_with_shap

from mlflow.tracking import MlflowClient

#  Configurations
SEED = 42
TARGET_COL = "converted"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

MLFLOW_EXPERIMENT = "lead_scoring_ml_experiment"
MLFLOW_MODEL_NAME = "Lead_Conversion_Prediction"
MLFLOW_MODEL_ARTIFACT_PATH = "best_model_pipeline"

''' Models and hyperparameter grids. Here using 5 Models namely random forest, logistic regression,
 gradient boosting, decision tree, knn classifier and corresponding parameters for each model used.
 The dictionary of models can be altered and additional models can be added according to the 
 training requirements.'''
MODELS = {
    "random_forest": RandomForestClassifier(random_state=SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    "decision_tree": DecisionTreeClassifier(random_state=SEED),
    "knn": KNeighborsClassifier(),
}

PARAMS = {
    "random_forest": {"clf__n_estimators": [50, 100]},
    "logistic_regression": {"clf__C": [1.0, 10.0]},
    "gradient_boosting": {"clf__n_estimators": [50, 100]},
    "decision_tree": {"clf__max_depth": [3, 5, 10]},
    "knn": {"clf__n_neighbors": [3, 5, 7]},
}

def normalize_column_name(col):
    return re.sub(r'[\s]+', '_', col.strip().lower())


def load_and_prepare_data():
    """
    Load raw lead scoring data from Postgres, then clean.

    Returns:
        df_clean (pd.DataFrame): Cleaned dataframe ready for modeling.
    """
    print("Loading data from PostgreSQL...")
    df_raw = read_table("lead_scoring")  # PostgreSQL table name

    print(f"Raw data shape: {df_raw.shape}")

    print("Performing basic cleaning...")
    df_clean = basic_cleaning(df_raw)
    df_clean.columns = [normalize_column_name(col) for col in df_clean.columns]

    print(f"Cleaned data shape: {df_clean.shape}")

    return df_clean

def phase_1_sample_prototype(df, sample_frac=0.2):
    """
    Phase 1: Prototype on a small fraction of data to speed up experiments.

    Args:
        df: Full dataframe raw loaded.
        sample_frac: Fraction (e.g. 0.2 for 20%) of data to sample.

    Returns:
        best_pipeline: Best pipeline found from sampled data.
    """
    print("Columns in df_clean:", df.columns.tolist())

    print("\n=== Phase 1: Prototype on sample data ===")

    df_sample = df.sample(frac=sample_frac, random_state=SEED)
    X = df_sample.drop(columns=[TARGET_COL])
    y = df_sample[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=SEED
    )
    print(f"Sample Train/Test shapes: {X_train.shape}, {X_test.shape}")

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}_phase1")
    best_auc = -np.inf
    best_pipeline = None

    with mlflow.start_run(run_name="phase1_sample_experiment") as prun:
        for model_name, model in MODELS.items():
            print(f"Training on sample: {model_name}")
            pipeline = Pipeline([
                ("preprocessing", get_preprocessing_pipeline()),
                ("clf", model)
            ])

            grid = GridSearchCV(
                pipeline,
                param_grid=PARAMS[model_name],
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            with mlflow.start_run(run_name=model_name, nested=True):
                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)
                y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else y_pred

                auc = roc_auc_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_params(grid.best_params_)
                mlflow.log_param("model", model_name)
                mlflow.log_metric("roc_auc", auc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", acc)

                mlflow.sklearn.log_model(grid.best_estimator_, artifact_path=MLFLOW_MODEL_ARTIFACT_PATH)

                print(f"{model_name} - ROC-AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_pipeline = grid.best_estimator_

    print(f"Phase 1 best ROC-AUC: {best_auc:.4f}")
    return best_pipeline


def phase_2_full_split(df):
    """
    Phase 2: Train/test split on full data, robust model selection.

    Args:
        df: Full dataframe raw loaded and cleaned.

    Returns:
        best_pipeline: Best pipeline fitted to 80% train data.
        best_run_id: MLflow run ID of best pipeline run.
    """
    print("\n=== Phase 2: Full data 80/20 train-test split for model selection ===")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=SEED
    )
    print(f"Train/Test split sizes: {X_train.shape}, {X_test.shape}")

    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}_phase2")
    best_auc = -np.inf
    best_pipeline = None
    best_run_id = None

    with mlflow.start_run(run_name="phase2_full_train_test") as main_run:
        for model_name, model in MODELS.items():
            print(f"Training on full split: {model_name}")
            pipeline = Pipeline([
                ("preprocessing", get_preprocessing_pipeline()),
                ("clf", model)
            ])

            grid = GridSearchCV(
                pipeline,
                PARAMS[model_name],
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            with mlflow.start_run(run_name=model_name, nested=True):
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)
                y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, "predict_proba") else y_pred

                auc = roc_auc_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_params(grid.best_params_)
                mlflow.log_param("model", model_name)
                mlflow.log_metric("roc_auc", auc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", acc)

                mlflow.sklearn.log_model(grid.best_estimator_, artifact_path=MLFLOW_MODEL_ARTIFACT_PATH)

                print(f"{model_name} - ROC-AUC: {auc:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

                if auc > best_auc:
                    best_auc = auc
                    best_pipeline = grid.best_estimator_
                    best_run_id = mlflow.active_run().info.run_id

    print(f"Phase 2 best ROC-AUC: {best_auc:.4f} (Run ID: {best_run_id})")
    return best_pipeline, best_run_id


def phase_3_retrain_full(df, best_pipeline, best_run_id):
    """
    Phase 3: Retrain the best pipeline on 100% of data for final deployment
    and register in MLflow Model Registry.

    Args:
        df: Full dataframe raw loaded and cleaned.
        best_pipeline: Best pipeline fitted on train split.
        best_run_id: MLflow run ID of best pipeline used for registration.
    """
    print("\n=== Phase 3: Retrain best model on 100% data ===")

    X_full = df.drop(columns=[TARGET_COL])
    y_full = df[TARGET_COL]

    best_pipeline.fit(X_full, y_full)  # Refit on full data

    # Save pipeline artifact
    ARTIFACT_NAME = ARTIFACT_DIR / "final_pipeline.pkl"
    joblib.dump(best_pipeline, ARTIFACT_NAME)
    print(f"✅ Best pipeline retrained and saved: {ARTIFACT_NAME}")

    # Register model in MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"

    '''
    existing_models = [rm.name for rm in client.list_registered_models()]
    if MLFLOW_MODEL_NAME not in existing_models:
        client.create_registered_model(MLFLOW_MODEL_NAME)'''

    def safe_register_model(Client, model_name):
        try:
            Client.create_registered_model(model_name)
        except Exception as e:
            # Likely model already exists
            print(f"Model '{model_name}' probably exists, skipping creation: {e}")

    safe_register_model(client, MLFLOW_MODEL_NAME)
    # Register new version
    version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=model_uri,
        run_id=best_run_id
    )
    print(f"✅ Registered model version {version.version} in MLflow Model Registry under '{MLFLOW_MODEL_NAME}'")


if __name__ == "__main__":
    # Load data once
    df_clean = load_and_prepare_data()

    '''
    #Following code can be used if working directly with csv files and not connected to PostgreSQL.
    df_raw = pd.read_csv("data/raw/Lead Scoring.csv")
    df_clean = basic_cleaning(df_raw)
'''
    # Phase 1: Prototype on small sample
    best_pipeline_phase1 = phase_1_sample_prototype(df_clean, sample_frac=0.2)

    # Phase 2: Full train-test split, select best model
    best_pipeline_phase2, run_id_phase2 = phase_2_full_split(df_clean)

    # After best_pipeline_phase2 and run_id_phase2 are returned

    # 🔍 Sample from cleaned full data before pipeline (with engineered features)
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]

    # Apply feature engineering & rare grouping to sample for SHAP
    from src.preprocessing.pipelines import engineering_features, rare_category_grouper

    X_sample = X.sample(n=100, random_state=SEED).copy()
    X_sample = engineering_features(X_sample)
    X_sample = rare_category_grouper(X_sample)

    # Call SHAP explainer with updated sample
    explain_model_with_shap(best_pipeline_phase2, X_sample)

    '''
    # SHAP explanation on best model
    X = df_clean.drop(columns=[TARGET_COL])
    _, X_test, _, _ = train_test_split(X, df_clean[TARGET_COL], stratify=df_clean[TARGET_COL], test_size=0.2,
                                       random_state=SEED)
    X_sample = X_test.sample(n=100, random_state=SEED)
    explain_model_with_shap(best_pipeline_phase2, X_sample)'''

    # Phase 3: Retrain best model on full data & register
    phase_3_retrain_full(df_clean, best_pipeline_phase2, run_id_phase2)

    print("\nAll phases completed!")





























'''import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.preprocessing.cleaning import basic_cleaning
from src.preprocessing.pipelines import (get_preprocessing_pipeline)
from mlflow.tracking import MlflowClient

# CONFIGS
SEED = 42
TARGET_COL = "Converted"
EXPERIMENT_NAME = "lead_scoring_ml_experiment"
ARTIFACT_DIR = Path("artifacts")

ARTIFACT_NAME = ARTIFACT_DIR / "final_lead_classifier.pkl"
MLFLOW_MODEL_ARTIFACT_PATH = "Best_model_pipeline"
MLFLOW_MODEL_NAME = "Lead_Conversion"

# MODELS and PARAMS
MODELS = {
    "random_forest": RandomForestClassifier(random_state=SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    "decision_tree": DecisionTreeClassifier(random_state=SEED),
    "knn": KNeighborsClassifier(),
}

PARAMS = {
    "random_forest": {"clf__n_estimators": [50, 100]},
    "logistic_regression": {"clf__C": [1.0, 10.0]},
    "gradient_boosting": {"clf__n_estimators": [50, 100]},
    "decision_tree": {"clf__max_depth": [3, 5, 10]},
    "knn": {"clf__n_neighbors": [3, 5, 7]},
}

# LOAD DATA
df = pd.read_csv("data/raw/Lead Scoring.csv")
df = basic_cleaning(df)

# SAMPLE 20% FOR QUICK EXPERIMENTS
df_sample = df.sample(frac=SAMPLE_FRAC, random_state=SEED)
X = df_sample.drop(columns=[TARGET_COL])
y = df_sample[TARGET_COL]

# SPLIT TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=SEED
)
print(X_train.shape, X_test.shape)

# SETUP MLflow
mlflow.set_experiment(EXPERIMENT_NAME)
best_Pipeline = None
best_auc = -np.inf
best_run_id = None

# TRAIN AND LOG MODELS
with mlflow.start_run(run_name="multi_model_experiment") as parent_run:
    for model_name, model in MODELS.items():
        print(f"\n🔄 Training model: {model_name}")

        # PIPELINE: preprocessing + model (step named 'clf')
        pipeline = Pipeline([
            ("preprocessing", get_preprocessing_pipeline()),
            ("clf", model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid=PARAMS[model_name],
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        with mlflow.start_run(run_name=model_name, nested=True):
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            y_proba = (
                grid.predict_proba(X_test)[:, 1]
                if hasattr(grid, "predict_proba")
                else y_pred
            )

            # METRICS
            auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)

            # LOG to MLflow
            mlflow.log_param("model", model_name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path=model_name)

            print(f"✅ {model_name} ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

            # CHECK BEST MODEL
            if auc > best_auc:
                best_auc = auc
                best_Pipeline = grid.best_estimator_
                best_run_id = mlflow.active_run().info.run_id

# SAVE BEST MODEL
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_Pipeline, ARTIFACT_NAME)
print(f"\n✅ Best Pipeline saved to {ARTIFACT_NAME}")
print(f"🏅 Best ROC-AUC: {best_auc:.4f} (run_id={best_run_id})")

print("\n📊 Run MLflow UI to explore:  mlflow ui")




def register_best_model(run_id):
    """Registers the best model pipeline logged in MLflow to the Model Registry."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{MLFLOW_MODEL_ARTIFACT_PATH}"

    # Check if model name exists, else create
    existing_models = [rm.name for rm in client.list_registered_models()]
    if MLFLOW_MODEL_NAME not in existing_models:
        client.create_registered_model(MLFLOW_MODEL_NAME)

    # Create new version in registry
    model_version = client.create_model_version(
        name=MLFLOW_MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    print(f"Registered Model '{MLFLOW_MODEL_NAME}' with version {model_version.version}")


# Register best model in MLflow Model Registry
register_best_model(best_run_id)
'''

'''
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV

# Importing Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from src.preprocessing.pipelines import get_preprocessing_pipeline
from src.preprocessing.cleaning import basic_cleaning
import joblib

# Configuration Variables for sampling

SEED = 42
SAMPLE_SIZE = 0.2  # 20%
EXPERIMENT_NAME = "Lead_Scoring_ML"

# Loading the Dataset and setting the target variables
df = pd.read_csv("data/raw/Lead Scoring.csv")
df = basic_cleaning(df)
target = "Converted"

# Perform Sampling on the dataset and Use only 20% of data for development/initial training
df_sample = df.sample(frac=SAMPLE_SIZE, random_state=SEED)
X = df_sample.drop(columns=[target])
y = df_sample[target]

# Split data into Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)  # 80% train, 20% test of sample

# A Dictionary of Models to Train and sleect upon
MODELS = {
    "random_forest": RandomForestClassifier(random_state=SEED),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    "decision_tree": DecisionTreeClassifier(random_state=SEED),
    "knn": KNeighborsClassifier(),
}

# A Dictionary of parameters for each associated model
PARAMS = {
    "random_forest": {"clf__n_estimators": [50, 100]},
    "logistic_regression": {"clf__C": [1.0, 10.0]},
    "gradient_boosting": {"clf__n_estimators": [50, 100]},
    "decision_tree": {"clf__max_depth": [3, 5, 10]},
    "knn": {"clf__n_neighbors": [3, 5, 7]}

}


# Setting up MLFlow
mlflow.set_experiment(EXPERIMENT_NAME)
best_model = None
best_score = -np.inf
best_run_id = None

with mlflow.start_run(run_name="multi_model_experiment") as main_run:
    for name, model in MODELS.items():
        # Pipeline: preprocessing + model
        pipe = make_pipeline(get_preprocessing_pipeline(), model)
        param_grid = PARAMS[name]
        search = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)

        # Start subrun for model
        with mlflow.start_run(run_name=name, nested=True):
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = (
                search.predict_proba(X_test)[:, 1]
                if hasattr(search, "predict_proba") else y_pred
            )
            score = roc_auc_score(y_test, y_proba)
            # Log params & metrics to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model", name)
            mlflow.log_metric("roc_auc", score)
            mlflow.log_metric("f1", f1_score(y_test, y_pred))
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.sklearn.log_model(search.best_estimator_, f"{name}_model")
            # Print current run info
            print(f"Model: {name}, ROC-AUC: {score:.3f}")
            if score > best_score:
                best_score = score
                best_model = search.best_estimator_
                best_run_id = mlflow.active_run().info.run_id

#  Save Final Best Model (For Full Data Retrain Later)
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)
best_model_path = artifacts_dir / "final_lead_classifier.pkl"
joblib.dump(best_model, best_model_path)
print(f"Best model (run_id={best_run_id}) saved to {best_model_path}")

print("\nView and compare all models/metrics in MLflow UI by running:\n  mlflow ui")

# ⭐ When ready, retrain on 100% data using only the best model & pipeline, then re-log in MLflow as "production" run and export new .pkl!
'''