import mlflow
import numpy as np

from fastapi import FastAPI, Response

from api import InferenceInput, InferenceOutput, PrepModelInput, PrepModelResponse
from config import ML_FLOW_URI, MODEL_CONFIG


mlflow.set_tracking_uri(ML_FLOW_URI)
models = {}

def load_model(model_uri):
    model_uri = MODEL_CONFIG['model']['model_uri']
    model = mlflow.pyfunc.load_model(model_uri)
    return model


app = FastAPI(
    title='Digit Classifier',
    description='Model classifies hand written digits, trained on MNIST dataset',
)

@app.get("/ping")
def ping():
    return Response(content='ok', media_type='text/plain')


@app.post('/internal/prep_model')
def prep_model(request: PrepModelInput):
    model_uri = request.model_uri
    models[model_uri] = load_model(model_uri)
    return PrepModelResponse(status=True)


@app.post('/infer')
def infer(request: InferenceInput):

    # Validate input
    # do something

    if request is None:
        return Response(content="Invalid request", media_type='text/plain')
    
    if request.model_uri not in models:
        return Response(content="Model not available", media_type='text/plain')
    
    model = models[request.model_uri]
    # Get input data in right format.
    raw_input = request.raw_image
    
    raw_input_reshaped = np.array(raw_input).reshape((1, 1, 28, 28)).astype(np.float32)

    pred = model.predict(raw_input_reshaped)

    class_id = np.argmax(pred[0])
    prob = np.max(pred[0])

    # return InferenceResponse(class_id=class_id,
    #                          prob=prob)
    return {
        "class_id" : int(class_id),
        "prob" : float(prob)
    }


