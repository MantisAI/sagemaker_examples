import os

from sagemaker.pytorch import PyTorch
import typer

def run_pytorch_sagemaker(data_path, role=os.environ.get("AWS_SAGEMAKER_ROLE"), instance_type="local", batch_size:int=16,
        epochs:int=5, learning_rate:float=1e-3, vocab_size:int=1000, seq_len:int=100, emb_size:int=100, hidden_size:int=200, num_layers:int=2):
    hyperparameters = {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "emb_size": emb_size,
        "hidden_size": hidden_size, 
        "num_layers": num_layers
    }
    
    pt = PyTorch(
        entry_point="train_pytorch.py",
        framework_version="1.9",
        py_version="py38",
        instance_type=instance_type,
        instance_count=1,
        role=role,
        hyperparameters=hyperparameters
    )
    pt.fit({"train": data_path})

if __name__ == "__main__":
    typer.run(run_pytorch_sagemaker)
