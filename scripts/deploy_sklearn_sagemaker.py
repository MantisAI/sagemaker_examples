import datetime
import os

from sagemaker.sklearn.model import SKLearnModel
import typer


def deploy_sklearn(
    model_path,
    instance_type="local",
    instance_count: int = 1,
    endpoint_name_prefix="sklearn",
    role=os.environ["AWS_SAGEMAKER_ROLE"],
):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"{endpoint_name_prefix}-{now}"
    print(f"Endpoint name: {endpoint_name}")

    sklearn = SKLearnModel(
        model_data=model_path,
        entry_point="src/predict_sklearn.py",
        framework_version="0.20.0",
        role=role
    )
    sklearn.deploy(
        initial_instance_count=instance_count, instance_type=instance_type, endpoint_name=endpoint_name
    )


if __name__ == "__main__":
    typer.run(deploy_sklearn)
