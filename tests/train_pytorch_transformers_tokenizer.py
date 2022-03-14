import os

from src.train_pytorch_transformers_tokenizer import train_pytorch


def test_train_pytorch(tmp_path):
    data_path = os.path.join(os.path.dirname(__file__), "data/")
    model_path = os.path.join(tmp_path, "model.pt")
    tokenizer_path = os.path.join(tmp_path, "tokenizer.pkl")
    train_pytorch(data_path, tmp_path)
    assert os.path.exists(model_path)
    assert os.path.exists(tokenizer_path)
