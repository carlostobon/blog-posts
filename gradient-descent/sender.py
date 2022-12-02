import httpx
from httpx import Headers

import glob
import os
import json
import base64

custom_dict = dict()

files = glob.glob('./images/*')
for file in files:
    with open(file, "rb") as binary_file:
        binary_data = binary_file.read()
        encoded = base64.b64encode(binary_data)
        to_send = encoded.decode('utf-8')
        custom_dict[os.path.basename(file)] = to_send

with open('index.md', 'r') as file:
    Lines = file.readlines()
    file.close()

markdown = ""

for line in Lines:
    markdown += line

cargo = {

    "title":"Gradient dent algorithm in python 1.42",

    "slug":"",

    "description":"The easiest way to understand the gradient descent algorithm \
        is coding it. Today I'll explain how it works and how to code it in Python.",

    "body": markdown,

    "keywords":"sgd machine learning, gradient descent machine learning, \
        gradient descent deep learning, gradient descent from scratch, sgd from scratch",

    "meta_description":"Understanding the gradient descent algorithm for \
        machine learning in python, an example using a simple function.",


    "category": ["Python", "DeepLearning", "Calculus", "MachineLearning"],

    "images": custom_dict,
}


r = httpx.post('http://127.0.0.1:8080/upload', json = cargo)
print(r.status_code)
print(r.json())



# ## REMOVE DIR
# # payload = {
    # "key": "zt4&sP&Z!6xnUw3txW2CG70r43OLW98M5UalZxw7w",
    # "slug": "first-post-in-my-life",
# }

# r = httpx.post('http://127.0.0.1:8080/remove', json = payload)

# print(r.status_code)
# print(r.text)
# print(r.json())





# print(dir(r))



# # Get method

# r = httpx.get("http://127.0.0.1:8080")
# print(r.status_code)
# print(r.json())
# print(r.text)
