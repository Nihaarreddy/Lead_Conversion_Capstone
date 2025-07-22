# src/preprocessing/etl_preprocessing.py

import os, joblib
import pandas as pd
import numpy as np
from pipelines import get_preprocessing_pipeline
import sqlalchemy
import argparse




def fetch_from_redshift():
    engine = sqlalchemy.create_engine(os.environ["REDSHIFT_CONN"])
    return pd.read_sql("SELECT * FROM lead_scoring", engine)

def main():
    df = fetch_from_redshift()
    pipe = get_preprocessing_pipeline()
    X = df.drop(columns=["Converted"])
    y = df["Converted"]

    # Fit pipeline on full data
    X_transformed = pipe.fit_transform(X)

    train_df = pd.DataFrame(X_transformed, columns=pipe.named_steps["preprocessor"].get_feature_names_out())
    train_df["Converted"] = y.values

    # Save processed data
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    train_df.to_csv("/opt/ml/processing/train/data.csv", index=False)

    joblib.dump(pipe, "/opt/ml/processing/preprocessor/pipeline.pkl")

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redshift-host", type=str)
    parser.add_argument("--redshift-user", type=str)
    parser.add_argument("--redshift-password", type=str)
    parser.add_argument("--redshift-db", type=str)
    args = parser.parse_args()

    host = args.redshift_host
    user = args.redshift_user
    password = args.redshift_password
    database = args.redshift_db

    # Now use psycopg2 or redshift_connector to connect
    main()
