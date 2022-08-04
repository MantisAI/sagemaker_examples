from requests import session
import json

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
import typer


def predict(text, endpoint_name):
    predictor = Predictor(endpoint_name=endpoint_name, serializer=JSONSerializer())
    prediction = predictor.predict({"text": text})
    print(prediction)


if __name__ == "__main__":
    typer.run(predict)
