from airflow import DAG
from datetime import datetime, timedelta
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.operators.python import PythonOperator


def run_preprocessing():
    print("✅ Preprocessing placeholder")

sagemaker_training_config = {
    "TrainingJobName": f"lead-conversion-job-{{{{ ts_nodash }}}}",
    "AlgorithmSpecification": {
        "TrainingImage": "763104351884.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        "TrainingInputMode": "File"
    },
    "RoleArn": "arn:aws:iam::766957562594:role/service-role/AmazonSageMaker-ExecutionRole-20250720T071710",
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://capstone-s3/raw-data/Batch_Lead_Set.csv",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "text/csv"
        }
    ],
    "OutputDataConfig": {
        "S3OutputPath": "s3://capstone-s3/model-artifacts/"
    },
    "ResourceConfig": {
        "InstanceType": "ml.m5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 30
    },
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 1800
    }
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="lead_conversion_mwaa_pipeline",
    default_args=default_args,
    description="Train Lead Conversion Model using SageMaker",
    schedule_interval="@daily",
    start_date=datetime(2025, 7, 22),
    catchup=False
) as dag:

    preprocessing_task = PythonOperator(
        task_id="run_preprocessing_etl",
        python_callable=run_preprocessing
    )

    training_task = SageMakerTrainingOperator(
        task_id="start_sagemaker_training",
        config=sagemaker_training_config,
        wait_for_completion=True
    )

    preprocessing_task >> training_task
