from evidently.report import Report
from evidently.metrics import DataDriftTable
from src.data_ingestion.db import read_table
from sklearn.model_selection import train_test_split

"""  Generate a drift report comparing  two datasets (FUll dataset and batch user dataset)
     from Postgres."""
def generate_drift_report_full_train_test(ref_table, test_table, report_path, ref_filter=None, test_filter=None):
    ref_df = read_table(ref_table)
    test_df = read_table(test_table)
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=ref_df, current_data=test_df)
    report.save_html(report_path)
    return report_path

'''   Generate a drift report comparing train and test splits from full data set'''
def generate_drift_report_split_train_test(full_table, report_path):
    full_df = read_table(full_table)
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)

    report = Report(metrics=[DataDriftTable()])  # Changed here
    report.run(reference_data=train_df, current_data=test_df)
    report.save_html(report_path)

    return report_path
