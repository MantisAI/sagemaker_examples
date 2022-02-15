import os

from sagemaker.huggingface import HuggingFace
import typer


def run_transformers_sagemaker(data_path, role=os.environ.get("AWS_SAGEMAKER_ROLE"), instance_type="local"):
    hyperparameters = {

    }
    hf = HuggingFace(
        entry_point="train_transformers.py",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        transformers_version="4.12",
        pytorch_version="1.9",
        py_version="py38",
        hyperparameters=hyperparameters
        
    )
    hf.fit({"train": data_path})

if __name__ == "__main__":
    typer.run(run_transformers_sagemaker) 
