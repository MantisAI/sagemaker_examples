import json
import os

from sagemaker.estimator import Estimator
import typer


def train_sagemaker(
    data_path,
    model_path,
    image_uri,
    entry_point=None,
    hyperparams=None,
    instance_type="local",
    instance_count: int = 1,
    role=os.environ["AWS_SAGEMAKER_ROLE"],
):
    if hyperparams:
        with open(hyperparams) as f:
            hyperparams = json.loads(f.read())

    model = Estimator(
        entry_point=entry_point,
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=instance_count,
        output_path=model_path,
        hyperparameters=hyperparams,
        role=role,
    )
    model.fit({"train": data_path})


if __name__ == "__main__":
    typer.run(train_sagemaker)
