import numpy as np
import requests
from pprint import pprint

raw_image = np.random.rand(28, 28).tolist()

input_request = {"raw_image": raw_image}

assert isinstance(input_request, dict)

response = requests.post(url="http://localhost:8080/infer/digit-classifier/dev", json=input_request)
print(response.status_code)
pprint(response.json())
