import httpx
from httpx import Headers

import glob
import os
import json
import base64
from PIL import Image

custom_dict = dict()


# Transform images format
target = ['jpg', 'png']
size = (900, 600)

images = glob.glob('./images/*')
for image in images:
    for ext in target:
        if ext in image:
            im = Image.open(image)
            out = im.resize(size, Image.LANCZOS)
            new_image = image.replace(ext, 'webp')
            out.save(new_image, format='webp')
            os.remove(image)



with open('info.json', 'r') as file:
    cargo = json.load(file)

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

cargo['body'] = markdown
cargo['images'] = custom_dict


r = httpx.post('https://wmatrix.xyz/server/upload', json = cargo)
# r = httpx.post('http://localhost:8080/upload', json = cargo)
print(r.status_code)
print(r.json())











# # remove dir
# payload = {
    # "key": "zt4&sP&Z!6xnUw3txW2CG70r43OLW98M5UalZxw7w",
    # "slug": "gradient-descent-algorithm-in-python",
# }

# r = httpx.post('https://www.deepmatrix.xyz/server/remove', json = payload)


# print(r.status_code)
# print(r.text)
# print(r.json())





# print(dir(r))



# # Get method

# r = httpx.get("http://127.0.0.1:8080")
# print(r.status_code)
# print(r.json())
# print(r.text)
