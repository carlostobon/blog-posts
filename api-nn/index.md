This post assumes you already have a trained [neural newtwork][1] using **pytorch**, I strongly recommend to
follow this [post][1] before you read it, however, in
case you trained it manually or using other approach, 
it's also possible to follow this tutorial, you just need to work on the details for 
your specific case.

Once you've trained a neural network, it's time to make it available for production, to get it done we'll be
creating an API with the framework **Flask** to serve the net. All you need to build this project is: 

[1]: </posts/neural-network-from-scratch-in-python>


- Net -> An already trained neural net.
- Scalers -> In case you scaled your data when training it.


<div class="special-ivory">
Flask is python is a micro-framework for web servers.
</div>


## Serializing The object

To be able to use the network we need to serialize it using **Pickle**, 
<em class="important-orange">serialize</em> means converting the object to a series of bytes, that way it can be
saved as we normally do with regular files and be used later once is needed.  


To serialize the [network][2] we've trained using **pytorch**:

[2]: </posts/neural-network-from-scratch-in-python> ""


```python
# ./neural_network.py
# torch needs to be already imported
# model = NeuraltNet() object
torch.save(model.state_dict(), './model.pkl')
```

Once the model is saved you should see a **model.pkl** file in the current directory,
let's save the scalers (in case you used any):

```python
# ./neural_network.py
# saving the scaler objects
with open('x_scaler.pkl', 'wb') as file:
   pickle.dump(x_scaler, file)

with open('y_scaler.pkl', 'wb') as file:
   pickle.dump(y_scaler, file)
```

Now your current directory should looks like:


```sh
.
└── current_directory/
    ├── neural_network.py
    ├── model.pkl
    ├── x_scaler.pkl
    └── y_scaler.pkl
```


## Setting up the work environment

As we already have the three objects needed, let's start working on the API, first of all
we got to create a new directory which we'll called <em class="important-teal">**api**</em>, get into that directory and run
the next commands to set up a python virtual environment.

```sh
mkdir api
cd api
python -m venv venv
. venv/bin/activate
```

Now we create a directory called **models** and a file **app.py**. In the first 
one we're going to place everything related
with the network so we should bring the three files (**network** and **scalers**) 
to this directory, and the second one will be the API.


Your files tree should looks like:

```sh
.
└── api/
  ├── app.py
  ├── venv/
  └── models/
      ├── model.pkl
      ├── x_scaler.pkl
      ├── y_scaler.pkl
      ├── setter.py
      └── __init__.py
```


If you see carefully we've created two more files, **setter** which we'll use to load the network and the scalers,
**__init__.py** allows us to treat **models** as a package, that way we can import all objects in there from the **app** file <em class="important-orange">app.py</em>, 
now let's install the needed packages to make it works. To install packages in python most 
people would use **pip** (hope you too), it should be done like this.

```sh
pip install flask numpy sklearn
```

## Loading the objects in the setter file

Let's begin importing the needed packages to load and process the network and scalers:


```python
# setter.py
import torch
import torch.nn as nn
import pickle
import numpy as np
```
Now we copy the **NeuralNet** class we used to build the network, in case you trained it 
manually just use your class


```python
# setter.py
class NeuralNet(nn.Module):
  def __init__(self):
      super(NeuralNet, self).__init__()
      self.linear_one = nn.Linear(4, 8)
      self.linear_two = nn.Linear(8, 4)
      self.linear_three = nn.Linear(4, 1)
      self.selu = nn.SELU()

  def forward(self, x):
      out = self.linear_one(x)
      out = self.selu(out)
      out = self.linear_two(out)
      out = self.selu(out)
      out = self.linear_three(out)
      return out
```

After creating the neural network class class, we got to generate an instance of that class and load
the weights:


```python
# setter.py
# your absolute path
path = 'your_path/api/models/'
model = NeuralNet()
model.load_state_dict(torch.load(path + 'model.pkl'))
```

Well the neural network is already mounted, let's load the scalers:

```python
# setter.py
with open(path + 'x_scaler.pkl', 'rb') as file:
   x_scaler = pickle.load(file)

with open(path + 'y_scaler.pkl', 'rb') as file:
   y_scaler = pickle.load(file)
```

Lastly, it's important to create a pipeline, the reason is we want to send a list
to the API with the corresponded data; so this pipeline will receive a list, 
transform it to tensors, then scale it and feed the network, once the network returns
an output (estimation for the inputs we've given to it) the pipeline makes an inverse escalation
an give us the output in the original format.

A brief explanation of what is the pipeline doing in the for us:

<img src="imageone.webp" alt="api flask neural network" />

Adding the **pipeline** to **setter** file:


```python
# setter.py
def pipeline(matrix: list) -> list:
 out = np.array(matrix)
 out = x_scaler.transform(out)
 out = torch.tensor(out, dtype=torch.float)
 out = model(out)
 out = out.detach().numpy()
 out = y_scaler.inverse_transform(out)
 out = out.tolist()
 return out
```

## Building the API

So far we've been building all things needed to make the API works as we want,
it was the hard work and practically the API is already done, let's now move to the
**app.py** file and finish the project.

let's import **Flask** and some of its functions, also the objects we've set in the **setter.py** file.


```python
# app.py
from flask import Flask, request, jsonify
from models.setter import model, pipeline,\
  x_scaler, y_scaler
```

Once we run the app.py file (our server), all those objects will load automatically,
now adding a **Flask** instance:


```python
# app.py
app = Flask(__name__)
```

let's add our unique route for this API <em class="important-teal">/</em>


```python
# app.py
@app.post('/')
def api():
 to_return = request.json
 to_return = pipeline(to_return)
 return jsonify(to_return)
```

Our route will be a **post** route, it means if you open it with a browser it wont return anything, 
this API will only work if we send data to it, 
this data must be a list (a serialized one using JSON), also such data must match with the network structure 
we're serving.


Adding the last line of code to make sure once we run the **app.py** file **python** runs 
the **Flask** API; Btw we'll be using the port $5000$.


```python
# app.py
if __name__ == "__main__":
  app.run(debug=False, port=5000)
```

The moment of truth, time to run the **app.py** file, in your terminal:

```sh
python app.py
```

It should returns something like this:

```sh
* Serving Flask app 'app' (lazy loading)
* Environment: production
 WARNING: This is a development server. Do not use it in a production deployment.
 Use a production WSGI server instead.
* Debug mode: on
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
* Restarting with stat
```

## Testing the API

Time to test the API, let's feed it with some samples (combinations in case you followed the other [tutorial][1].):


```sh
[10, 10, 10, 10], [8, 9, 10, 9], [7, 10, 8, 12], [11, 7, 6, 14]
```
To get such estimation, we can use the library **requests** in python:


```python
# testing.py
import requests
url = 'http://127.0.0.1:5000/'
payload = [[10, 10, 10, 10],
         [8, 9, 10, 9],
         [7, 10, 8, 12],
         [11, 7, 6, 14]]

res = requests.post(url, json = payload)
print(res.json())
```
You should get something like this, not equal but quite similar.

```sh
[[402.5483703613281], [162.580810546875], [95.20247650146484], [144.662109375]]
```

Well that's it, everything is working as expected, now you can deploy it using your preferred method. Hope
you've enjoyed and learned something new from this post.


