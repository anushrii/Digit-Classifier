from typing import List
from pydantic import BaseModel


class InferenceInput(BaseModel):
    """
    InferenceInput is a Pydantic model that represents the input data required for an inference request.

    Attributes:
        model_uri (str): The URI of the model to be used for inference.
        raw_image (List[List[float]]): A 2D list representing the raw image data to be processed.
    """

    model_uri: str
    raw_image: List[List[float]]


class InferenceOutput(BaseModel):
    """
    InferenceOutput is a Pydantic model that represents the output of an inference operation.

    Attributes:
        class_id (int): The identifier for the predicted class.
        prob (float): The probability score of the prediction.
    """

    class_id: int
    prob: float


class PrepModelInput(BaseModel):
    """
    PrepModelInput is a Pydantic model used to prepare model.

    Attributes:
        model_uri (str): The URI of the model to be prepared.
    """

    model_uri: str


class PrepModelResponse(BaseModel):
    """
    PrepModelResponse is a Pydantic model used to represent the response of a model preparation request.

    Attributes:
        status (bool): Indicates whether the model preparation was successful.
    """

    status: bool
