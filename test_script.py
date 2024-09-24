import numpy as np
import requests
from pprint import pprint

model_uri = "models:/digit-classifier@dev"

raw_image = np.random.rand(28, 28).tolist()

input_request = {
    'model_uri' : model_uri,
    'raw_image' : raw_image
}

assert isinstance(input_request, dict)

response = requests.post(url="http://localhost:8080/infer",json=input_request)
print(response.status_code)
pprint(response.json())
