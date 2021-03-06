import datetime
import os

from sagemaker.huggingface import HuggingFace
import typer


def run_transformers_sagemaker(
    data_path,
    model_path,
    job_name_prefix="transformers",
    role=os.environ.get("AWS_SAGEMAKER_ROLE"),
    instance_type="local",
    pretrained_model="bert-base-uncased",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
    weight_decay: float = 0.01,
):
    hyperparameters = {
        "pretrained_model": pretrained_model,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
    }
    hf = HuggingFace(
        entry_point="src/train_transformers.py",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version="4.12",
        pytorch_version="1.9",
        py_version="py3",
        hyperparameters=hyperparameters,
        output_path=model_path,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{job_name_prefix}-{now}"
    print(f"Job name: {job_name}")

    hf.fit({"train": data_path}, job_name=job_name)


if __name__ == "__main__":
    typer.run(run_transformers_sagemaker)
