#!/usr/bin/env python
import json

from pydantic import BaseModel

from train_sklearn import train

class Hyperparams(BaseModel):
    min_df: int = 5
    max_ngrams: int = 1
    stopwords="english"
    loss="hinge"
    learning_rate: float = 1e-4    

if __name__ == "__main__":
    data_path = "/opt/ml/input/data/train"
    model_path = "/opt/ml/model/"
    hyperparams_path = "/opt/ml/input/config/hyperparameters.json"

    with open(hyperparams_path) as f:
        hyperparams_raw = json.loads(f.read())
        hyperparams = Hyperparams(**hyperparams_raw)
    print(hyperparams)
    
    train(
        data_path,
        model_path,
        **hyperparams.dict()
    )
