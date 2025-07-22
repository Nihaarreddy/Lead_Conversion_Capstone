# Importing all the libraries and methods

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder,FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

# This function loads content present in the config/config.yaml file which is categories of features
def load_config(config_path=Path(__file__).parents[1] / "config" / "config.yaml"):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg["features"]

# Function to map Yes/No to 1/0
def yes_no_map(X):
        return np.where((X == 'Yes') | (X == 1), 1, 0).astype(int)

def engineering_features(X):
    X = X.copy()
    X["visits_per_page"] = X["totalvisits"] / (X["page_views_per_visit"] + 1)
    X["activity_score_ratio"] = X["asymmetrique_activity_score"] / (X["asymmetrique_profile_score"] + 1)
    X["specialization_missing"] = X["specialization"].isna().astype(int)
    X["is_lead_profile_unknown"] = X["lead_profile"].isna().astype(int)
    return X





#  combines infrequent categories, reducing dimensionality and improving generalization.
def rare_category_grouper(X):
    X = X.copy()
    threshold = 100
    rare_cols = ["lead_source", "specialization", "tags", "city", "country"]
    for col in rare_cols:
        freq = X[col].value_counts()
        common_cats = freq[freq >= threshold].index
        X.loc[~X[col].isin(common_cats), col] = "Other"
    return X

# This function creates multiple pipelines and combines them using ColumnTransformer
def get_preprocessing_pipeline():
    cfg = load_config()

    numeric_features = cfg["numeric"]
    categorical_features = cfg["categorical"]
    binary_features = cfg["binary"]
    drop_features = cfg.get("drop", [])




    # Numeric Transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    # Categorical Transformer
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Binary Transformer
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='No')),  # handle missing as 'No'
        ('map', FunctionTransformer(yes_no_map, validate=False)),
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features + ["visits_per_page", "activity_score_ratio"]),
        ("cat", categorical_transformer, categorical_features + ["specialization_missing", "is_lead_profile_unknown"]),
        ("bin", binary_transformer, binary_features),
        ("dropper", "drop", drop_features)
    ],
    remainder='drop'
    )
    # Create pipeline combining all
    pipe =  Pipeline([
        # ---- Feature Engineering ----
        ("feature_eng", FunctionTransformer(engineering_features)),

        # ---- Group rare categories BEFORE encoding ----
        ("rare_group", FunctionTransformer(rare_category_grouper)),
        ("preprocessor", preprocessor),
        # Feature selection steps:
        ("variance", VarianceThreshold(threshold=0.01))
    ]
    )

    return pipe


