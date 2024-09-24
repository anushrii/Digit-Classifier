import mlflow
import numpy as np

from fastapi import FastAPI, Response
from prometheus_fastapi_instrumentator import Instrumentator

from api import InferenceInput, InferenceOutput
from config import ML_FLOW_URI, MODEL_PREFIX


mlflow.set_tracking_uri(ML_FLOW_URI)
models = {}


def load_model(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    return model


app = FastAPI(
    title="Digit Classifier",
    description="Model classifies hand written digits, trained on MNIST dataset",
)
# default prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.get("/ping")
def ping():
    return Response(content="ok", media_type="text/plain")


@app.post("/internal/prep_model/{model_id}/{tag}")
def prep_model(model_id, tag):
    model_uri = MODEL_PREFIX + model_id + '@' + tag
    models[model_uri] = load_model(model_uri)
    return Response(content="ok", media_type="text/plain")


@app.post("/infer/{model_id}/{tag}")
def infer(model_id, tag, request: InferenceInput):

    # Validate inputs.
    if request is None:
        return Response(content="Invalid request", media_type="text/plain")

    model_uri = MODEL_PREFIX + model_id + '@' + tag
    if model_uri not in models:
        return Response(content="Model not available", media_type="text/plain")

    if request.raw_image is None:
        return Response(content="Invalid image data", media_type="text/plain")
    
    if len(request.raw_image) != 28 or len(request.raw_image[0]) != 28:
        return Response(content="Invalid image size", media_type="text/plain")

    model = models[model_uri]

    # Get input data in right format.
    raw_input = request.raw_image
    raw_input_reshaped = np.array(raw_input).reshape((1, 1, 28, 28)).astype(np.float32)

    pred = model.predict(raw_input_reshaped)
    class_id = np.argmax(pred[0])

    # convert logits to probabilities
    exp_preds = np.exp(pred[0])
    prob_preds = exp_preds / np.sum(exp_preds)
    prob = np.max(prob_preds)

    return InferenceOutput(class_id=int(class_id), prob=float(prob))
