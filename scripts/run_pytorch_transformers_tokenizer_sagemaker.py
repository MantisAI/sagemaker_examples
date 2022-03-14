import datetime
import os

from sagemaker.pytorch import PyTorch
import typer


def run_pytorch_sagemaker(
    data_path,
    model_path,
    job_name_prefix="pytorch",
    role=os.environ.get("AWS_SAGEMAKER_ROLE"),
    instance_type="local",
    batch_size: int = 16,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    seq_len: int = 100,
    emb_size: int = 100,
    hidden_size: int = 200,
    num_layers: int = 2,
):
    hyperparameters = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seq_len": seq_len,
        "emb_size": emb_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }
    environment = {"SAGEMAKER_REQUIREMENTS": "requirements.txt"}
    pt = PyTorch(
        entry_point="train_pytorch_transformers_tokenizer.py",
        source_dir="src/",
        environment=environment,
        framework_version="1.9",
        py_version="py38",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        hyperparameters=hyperparameters,
        output_path=model_path,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{job_name_prefix}-{now}"
    print(f"Job name: {job_name}")

    pt.fit({"train": data_path}, job_name=job_name)


if __name__ == "__main__":
    typer.run(run_pytorch_sagemaker)
