import os

from sagemaker.sklearn.model import SKLearnModel
import typer


def deploy_sklearn(
    model_path,
    instance_type="local",
    instance_count: int = 1,
    role=os.environ["AWS_SAGEMAKER_ROLE"],
):
    sklearn = SKLearnModel(
        model_data=model_path,
        entry_point="src/predict_sklearn.py",
        framework_version="0.20.0",
        role=role,
    )
    sklearn.deploy(initial_instance_count=instance_count, instance_type=instance_type)


if __name__ == "__main__":
    typer.run(deploy_sklearn)
