import os

from src.train_transformers import train_transformers


def test_train_pytorch(tmp_path):
    data_path = os.path.join(os.path.dirname(__file__), "data/")
    model_path = os.path.join(tmp_path, "pytorch_model.bin")
    train_transformers(
        data_path, tmp_path, pretrained_model="distilbert-base-uncased", epochs=1
    )
    assert os.path.exists(model_path)
