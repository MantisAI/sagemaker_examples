import os

from src.train_sklearn_naive_bayes import train


def test_train_sklearn(tmp_path):
    data_path = os.path.join(os.path.dirname(__file__), "data/")
    model_path = os.path.join(tmp_path, "model.pkl")
    train(data_path, tmp_path)
    assert os.path.exists(model_path)
