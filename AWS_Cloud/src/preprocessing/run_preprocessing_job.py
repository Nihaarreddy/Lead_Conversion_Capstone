from sagemaker.processing import ScriptProcessor, ProcessingOutput
import sagemaker

role = sagemaker.get_execution_role()
session = sagemaker.Session()

processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.ap-south-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    sagemaker_session=session,
)

processor.run(
    code="../src/preprocessing/etl_preprocessing.py",
    inputs=[],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train", destination="s3://capstone-s3/processed/train/"),
        ProcessingOutput(source="/opt/ml/processing/preprocessor", destination="s3://capstone-s3/preprocessor/")
    ],
    job_name="lead-scoring-preprocess-final"
)
