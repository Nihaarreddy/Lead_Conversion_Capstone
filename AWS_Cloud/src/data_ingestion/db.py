
# src/data_ingestion/db.py



# 📁 src/data_ingestion/db.py

import pandas as pd

def read_redshift_table(
    s3_path='s3://capstone-s3/raw-data/Batch_Lead_Set.csv',
    **kwargs
) -> pd.DataFrame:
    """
    Reads a data file from Amazon S3 (CSV format).

    Parameters:
    ----------
    s3_path : str
        Full S3 path to the .csv file. Example: 's3://your-bucket/path/file.csv'

    kwargs : dict
        Additional arguments to pass to pd.read_csv(), e.g., sep, encoding, etc.

    Returns:
    -------
    pandas.DataFrame
        DataFrame containing the contents of the S3 file.
    """

    print(f"📥 Reading data from redshift: {s3_path}")
    df = pd.read_csv(s3_path, **kwargs)
    print(f"✅ Loaded dataframe shape: {df.shape}")
    return df



