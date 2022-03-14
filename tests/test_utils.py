import os

from src.utils import load_data


def test_load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data/movie.csv")

    X, y = load_data(data_path)
    assert len(X) == 9
