from sagemaker import Model
import typer
import os


def deploy(
    model_path,
    image_uri,
    entry_point=None,
    instance_type="local",
    instance_count: int = 1,
    role=os.environ["AWS_SAGEMAKER_ROLE"],
):
    model = Model(
        model_data=model_path, entry_point=entry_point, image_uri=image_uri, role=role
    )
    predictor = model.deploy(
        instance_type=instance_type, initial_instance_count=instance_count
    )
    print(predictor.endpoint_name)


if __name__ == "__main__":
    typer.run(deploy)
