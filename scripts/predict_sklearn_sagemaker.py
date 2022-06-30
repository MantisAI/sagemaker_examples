from requests import session
import json

from sagemaker.sklearn.model import SKLearnPredictor
from sagemaker.serializers import JSONSerializer
import typer


def predict_sklearn(text, endpoint_name):
    sklearn = SKLearnPredictor(endpoint_name=endpoint_name, serializer=JSONSerializer())
    prediction = sklearn.predict({"text": text})
    print(prediction)


if __name__ == "__main__":
    typer.run(predict_sklearn)
