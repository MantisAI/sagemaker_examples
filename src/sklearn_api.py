#!/usr/bin/env python
import pickle
import os

from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()

class Data(BaseModel):
    text: str

@app.post("/invocations")
def predict(data: Data):
    model_path = os.environ.get("SM_MODEL_DIR", "models/")
    with open(os.path.join(model_path, "model.pkl"), "rb") as f:
        model = pickle.loads(f.read())
    y = model.predict([data.text])
    return y.tolist()

@app.get("/ping")
def health():
    return "\n"

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("sklearn_api:app", host="0.0.0.0", port=8080, log_level="info")    
