import datetime
import os

from sagemaker import Model
import typer


def deploy(
    model_path,
    image_uri,
    entry_point=None,
    instance_type="local",
    instance_count: int = 1,
    endpoint_name_prefix="endpoint",
    role=os.environ["AWS_SAGEMAKER_ROLE"],
):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    endpoint_name = f"{endpoint_name_prefix}-{now}"
    print(f"Endpoint name: {endpoint_name}")

    model = Model(
        model_data=model_path, entry_point=entry_point, image_uri=image_uri, role=role
    )
    model.deploy(
        instance_type=instance_type, initial_instance_count=instance_count, endpoint_name=endpoint_name
    )


if __name__ == "__main__":
    typer.run(deploy)
