import pandas as pd
from pathlib import Path
import yaml
import numpy as np

# This function loads content present in the config/config.yaml file which is categories of features
def load_config(config_path=Path(__file__).parents[1] / "config" / "config.yaml"):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg["features"]

# Performing different data cleaning steps like dropping irrelevant features, replacing placeholders with NaN, etc
def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cfg = load_config()

    # Replace placeholders like "Select", "", "Not Updated" with NaN
    df.replace(["Select", "Not Updated", "", "nan", "NaN"], np.nan, inplace=True)

    return df
