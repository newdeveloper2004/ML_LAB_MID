import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

app = FastAPI()
reg = joblib.load(os.path.join(ROOT, "models", "model1.pkl"))
clf = joblib.load(os.path.join(ROOT, "models", "model2.pkl"))


class RegressionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_width: float


class ClassificationInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "index.html"))


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    X = pd.DataFrame([data.model_dump()])
    return {"result": round(float(reg.predict(X)[0]), 3)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    X = pd.DataFrame([data.model_dump()])
    return {"result": str(clf.predict(X)[0])}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
