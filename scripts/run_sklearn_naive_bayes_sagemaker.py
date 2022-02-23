import datetime
import os

from sagemaker.sklearn import SKLearn
import typer


def run_sklearn_naive_bayes_sagemaker(
    data_path,
    model_path,
    job_name_prefix="sklearn",
    instance_type="local",
    role=os.environ.get("AWS_SAGEMAKER_ROLE"),
    min_df: int = 5,
    max_ngram: int = 1,
    stopwords="english",
    alpha:float=1,
    fit_prior: bool = True,
):
    hyperparameters = {
        "min_df": min_df,
        "max_ngram": max_ngram,
        "stopwords": stopwords,
        "alpha": alpha,
        "fit_prior": fit_prior,
    }
    sk = SKLearn(
        entry_point="train_sklearn_naive_bayes.py",
        source_dir="src/",
        framework_version="0.20.0",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        hyperparameters=hyperparameters,
        output_path=model_path,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{job_name_prefix}-{now}"
    print(f"Job name: {job_name}")

    sk.fit({"train": data_path}, job_name=job_name)


if __name__ == "__main__":
    typer.run(run_sklearn_naive_bayes_sagemaker)
