# src/explainability/shap_explainer.py

# src/explainability/shap_explainer.py

import shap
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_names_from_column_transformer(column_transformer):
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]  # get final transformer

        try:
            names = transformer.get_feature_names_out(columns)
        except AttributeError:
            names = columns  # fallback

        feature_names.extend(names)

    return feature_names


def extract_column_transformer(preprocessor):
    """Extract ColumnTransformer from pipeline or return as is"""
    if isinstance(preprocessor, Pipeline):
        for name, step in preprocessor.steps:
            if isinstance(step, ColumnTransformer):
                return step
    elif isinstance(preprocessor, ColumnTransformer):
        return preprocessor
    raise ValueError("No ColumnTransformer found in preprocessing step.")


def explain_model_with_shap(pipeline, X_sample, output_dir="artifacts/shap"):
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 Extracting model and preprocessed sample...")

    model = pipeline.named_steps["clf"]
    preprocessing = pipeline.named_steps["preprocessing"]

    # Apply preprocessing pipeline manually to get the preprocessed input
    X_transformed = preprocessing.transform(X_sample)

    # Extract ColumnTransformer to get feature names
    column_transformer = extract_column_transformer(preprocessing)
    feature_names = get_feature_names_from_column_transformer(column_transformer)

    # Apply VarianceThreshold if used
    if "variance" in pipeline.named_steps:
        selector = pipeline.named_steps["variance"]
        support_mask = selector.get_support()
        X_transformed = X_transformed[:, support_mask]
        feature_names = [name for i, name in enumerate(feature_names) if support_mask[i]]

    print("⚙️ Generating SHAP explainer...")

    # Use KernelExplainer if model is not tree-based
    explainer = shap.Explainer(model.predict, X_transformed)
    shap_values = explainer(X_transformed)

    # Summary plot
    summary_path = Path(output_dir) / "shap_summary_plot.png"
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    print(f"✅ SHAP summary plot saved to {summary_path}")

    # Bar plot
    bar_path = Path(output_dir) / "shap_bar_plot.png"
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(f"✅ SHAP bar plot saved to {bar_path}")


'''
import shap
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_names_from_column_transformer(column_transformer):
    feature_names = []

    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]  # get the final transformer

        try:
            names = transformer.get_feature_names_out(columns)
        except AttributeError:
            names = columns  # fallback to original column names

        feature_names.extend(names)

    return feature_names


def explain_model_with_shap(pipeline, X_sample, output_dir="artifacts/shap"):
    """
    Generate SHAP explanation plots for a trained pipeline and a sample of input data.

    Args:
        pipeline: Trained sklearn pipeline (preprocessing + model).
        X_sample: Sample dataframe used to calculate SHAP values.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("🔍 Extracting model and preprocessed sample...")
    model = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["preprocessing"]
    if isinstance(preprocessor, Pipeline):
        for name, step in preprocessor.steps:
            if isinstance(step, ColumnTransformer):
                preprocessor = step
                break

    # Transform features using preprocessing pipeline
    X_transformed = preprocessor.transform(X_sample)

    # Get feature names after preprocessing
    
    try:
        cat_features = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
    except:
        cat_features = []

    feature_names = (
        preprocessor.named_transformers_['num'].feature_names_in_.tolist()
        + list(cat_features)
        + preprocessor.named_transformers_['bin'].feature_names_in_.tolist()
    )
    # Get feature names safely from the original DataFrame
    feature_names = get_feature_names_from_column_transformer(preprocessor)

    # If VarianceThreshold is used
    if "variance" in pipeline.named_steps:
        selector = pipeline.named_steps["variance"]
        support_mask = selector.get_support()
        X_transformed = X_transformed[:, support_mask]
        feature_names = [name for idx, name in enumerate(feature_names) if support_mask[idx]]

    print("⚙️ Generating SHAP explainer...")

    explainer = shap.Explainer(model.predict, X_transformed)

    shap_values = explainer(X_transformed)

    # Summary plot
    summary_path = Path(output_dir) / "shap_summary_plot.png"
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path)
    plt.close()
    print(f"✅ SHAP summary plot saved to {summary_path}")

    # Bar plot
    bar_path = Path(output_dir) / "shap_bar_plot.png"
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
    print(f"✅ SHAP bar plot saved to {bar_path}")
'''