import random
import requests

# Generate a 28x28 list of lists of random floats
raw_image = [[random.random() for _ in range(28)] for _ in range(28)]

input_request = {"raw_image": raw_image}

assert isinstance(input_request, dict)

response = requests.post(url="http://localhost:8080/infer/digit-classifier/dev", json=input_request)
print(response.status_code)
print(response.json())