# generate_drift_reports.py
from src.monitoring.drifts import generate_drift_report_full_train_test, generate_drift_report_split_train_test

def main():
    '''
        Call the Two reports to be generated
    '''
    generate_drift_report_full_train_test(
        ref_table="lead_scoring",
        test_table="user_batch_upload",
        report_path="reports/drift_full_vs_batch.html"
    )
    generate_drift_report_split_train_test(
        full_table="lead_scoring",
        report_path="reports/drift_train_test.html"
    )
    print("Drift reports generated.")

if __name__ == "__main__":
    main()
