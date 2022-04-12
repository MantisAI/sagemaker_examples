import argparse
import pickle
import json
import os


def model_fn(model_path):
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.loads(f.read())
    return model


def input_fn(input_data, content_type="application/json"):
    data = json.loads(input_data)
    return [data["text"]]


def predict_sklearn(text, model_path):
    model = model_fn(model_path)
    y = model.predict([text])
    print(y)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("text", type=str)
    argparser.add_argument("--model_path", type=str)
    args = argparser.parse_args()

    predict_sklearn(args.text, args.model_path)
