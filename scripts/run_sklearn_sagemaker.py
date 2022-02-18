import datetime
import os

from sagemaker.sklearn import SKLearn
import typer


def run_sklearn_sagemaker(
    data_path,
    model_path,
    job_name_prefix="sklearn",
    instance_type="local",
    role=os.environ.get("AWS_SAGEMAKER_ROLE"),
    min_df: int = 5,
    max_ngram: int = 1,
    stopwords="english",
    loss="hinge",
    learning_rate: float = 1e-4,
):
    hyperparameters = {
        "min_df": min_df,
        "max_ngram": max_ngram,
        "stopwords": stopwords,
        "loss": loss,
        "learning_rate": learning_rate,
    }
    sk = SKLearn(
        entry_point="src/train_sklearn.py",
        framework_version="0.20.0",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        hyperparameters=hyperparameters,
        output_path=model_path,
    )
    job_name = (
        f"{job_name_prefix}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )
    print(f"Job name: {job_name}")
    sk.fit({"train": data_path}, job_name=job_name)


if __name__ == "__main__":
    typer.run(run_sklearn_sagemaker)
