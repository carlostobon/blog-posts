When you're getting into the deep-learning field, it's quite important to understand how neural networks work, many of the mathematical models that you'll find in AI
are based in neural networks, therefore, learning how to code them could be
crucial to make your journey smoother.


Most time you will find tutorials explaining how to build neural networks for
classification problems (e.g. cat and dog), however, when you're a scientist or
data analyst, it's much more common to find yourself needing to tackle a
regression kind problem.

## The challenge
Let's suppose that you work for the **Company-XYZ**, this company is in the business
of growing rice, as for many companies out there, implementing new strategies 
to make more money is key to survive the market. Leaders in the company believe
the only way to achieve it is increasing the production or at least making
rice grow faster.


The research department begins testing **four** new type of 
<em class="important-orange">nutrients</em> to
improve the growing conditions, in that way, the department creates 
a dataset with $1000$ experiments that looks like:

<div class="table">
  <table>
    <thead>
      <tr>
        <th>Instance</th>
        <th>Nut_0</th>
        <th>Nut_1</th>
        <th>Nut_2</th>
        <th>Nut_3</th>
        <th>Total Production</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>12</td>
        <td>16</td>
        <td>8</td>
        <td>2</td>
        <td>100</td>
      </tr>
      <tr>
        <th>1</th>
        <td>8</td>
        <td>12</td>
        <td>22</td>
        <td>25</td>
        <td>120</td>
      </tr>
      <tr>
        <th>2</th>
        <td>8</td>
        <td>9</td>
        <td>18</td>
        <td>3</td>
        <td>94</td>
      </tr>
      <tr>
        <th>3</th>
        <td>16</td>
        <td>14</td>
        <td>9</td>
        <td>23</td>
        <td>98</td>
      </tr>
    </tbody>
  </table>
</div>

This table is showing us just the first four instances, 
but keep in mind that we've a thousand of them in the dataset, 
the first row (instance) means that a
combination of four nutrients ($12$, $16$, $8$, $2$) led to produce $100$ tons of rice, the same logic apply for the rest of rows.

## Importing libraries
Since we don't have a real dataset and no company wants to share that
kind of information, we need to create one to simulate this
scenario, let's start importing the libraries for this project.

```python
# Libraries needed to build the network
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.functional as F
```

## Building the dataset
```python
# Seed
RANDOM_STATE = 44

# Data creation
features, target = make_regression(
    n_samples = 1000, n_features = 4,
    noise = 0.05, random_state = RANDOM_STATE)

features += 10
target += 400
```

First of all, we import the **make_regression** function which returns a tuple with
two objects: <em class="important-orange">features</em> represents a matrix with four columns that will simulate the nutrients and <em class="important-orange">target</em>the total production.

**n_samples** means we want $1000$ instances, **n_features** is the number of explanatory features in the $X$ matrix (nutrients), **noise** is just a standard deviation we apply to the **target** to make the simulation more real, and **random_state** is the seed to generate always the same pseudo-random numbers.

If you see carefully we're adding $10$ units to **features** and $400$ to the **target**, it help us to make the dataset similar enough to the one explained in the example (rice production).


<div class="table">
  <table>
    <thead>
      <tr>
        <th>Instance</th>
        <th>Nut_0</th>
        <th>Nut_1</th>
        <th>Nut_2</th>
        <th>Nut_3</th>
        <th>Total Production</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>18</th>
        <td>10.2747</td>
        <td>9.7597</td>
        <td>10.8158</td>
        <td>9.3212</td>
        <td>422.8143</td>
      </tr>
      <tr>
        <th>68</th>
        <td>9.2724</td>
        <td>9.9714</td>
        <td>10.2236</td>
        <td>9.8019</td>
        <td>362.5996</td>
      </tr>
      <tr>
        <th>256</th>
        <td>9.2974</td>
        <td>8.9231</td>
        <td>9.4077</td>
        <td>9.7058</td>
        <td>290.8020</td>
      </tr>
      <tr>
        <th>340</th>
        <td>9.4361</td>
        <td>11.4212</td>
        <td>8.4001</td>
        <td>10.2041</td>
        <td>293.2563</td>
      </tr>
      <tr>
        <th>220</th>
        <td>9.6875</td>
        <td>10.4677</td>
        <td>9.5575</td>
        <td>10.1301</td>
        <td>367.5820</td>
      </tr>
      <tr>
        <th>803</th>
        <td>7.9646</td>
        <td>9.3005</td>
        <td>11.7100</td>
        <td>12.6046</td>
        <td>562.4164</td>
      </tr>
      <tr>
        <th>999</th>
        <td>9.9302</td>
        <td>10.2024</td>
        <td>10.9038</td>
        <td>9.5738</td>
        <td>431.5878</td>
      </tr>
    </tbody>
  </table>
</div>


This table is showing just a  couple of rows, but if you use the same **random_seed** used in this tutorial, those values should be identical to yours; **features** size should be $(1000, 4)$ 
and **target** $(1000,)$. The only thing left is reshaping the target vector to be $(1000, 1)$.

```python
target = target.reshape(-1, 1)
```

## Splitting the data
The next thing we need to do is to split the dataset, it means taking something like $80\%$ of
data to train the network, and the rest $20\%$ for validation and testing.


<img src="imageone.svg" alt="dataset split sample for training neural network" />


To split the dataset we'll be using the **Sklearn**'s splitter:

```py
# Splitting dataset
X_train, X_valid, y_train, y_valid = train_test_split(
  features, target, test_size=0.2, random_state=RANDOM_STATE)

X_valid, X_test, y_valid, y_test = train_test_split(
  X_valid, y_valid, test_size=0.5, random_state=RANDOM_STATE)
```

Since we're using <em class="important-teal">Pytorch</em> to create the network,
we must transform our arrays to <em class="important-orange">scaled tensors</em> (**Pytorch** objects), it will
let us to work with **Pytorch** and track the derivatives (gradients) 
throughout the calculations, it's needed to do the back-propagation 
section (we'll see more about it later).

<div class="special-ivory">
Pytorh is a Python library to create AI
</div>

So let's begin creating the scalers (data is not scaled yet):


## Scaling the data
```python
# scaler (it gets and holds the mean & std of X_train and y_train)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
```

This two objects will save $\sigma$ and $\mu$ from **X_train** and **y_train**,
and will be used to standardize each of all three samples like this:



<div class="math-div">

$$
X_{t} (\mu_{x\_t}, \sigma_{x\_t}) \longrightarrow \hspace{1em} Z = {X_i - \mu_{x\_t} \over \sigma_{x\_t}} 
$$

</div> 


<div class="math-div">

$$
y_{t} (\mu_{y\_t}, \sigma_{y\_t}) \longrightarrow \hspace{1em} Z = {y_i - \mu_{y\_t} \over \sigma_{y\_t}} 
$$

</div> 



## Transformer function
Once the scalers are created, we can scale the data, however, 
as transforming the samples to tensors is needed as well, it's much
better to create a sample that transform and scale the data, a good
approach could be this one:


```python
def transformer(inputs: list, scaler) -> list:
  to_return = list()
  for item in inputs:
    scaled = scaler.transform(item)
    scaled = torch.tensor(scaled, dtype=torch.float)
    to_return.append(scaled)
  return to_return
```

This function named **transformer** will expect a scaler object which contains $\mu$ and $\sigma$ of **X_train** or **y_train** and a list of samples, to make it more clear let's use it.


```python
# Transforming the datasets
X_train, X_test, X_valid = transformer(
  [X_train, X_test, X_valid], x_scaler)

y_train, y_test, y_valid = transformer(
  [y_train, y_test, y_valid], y_scaler)
```


## The neural network 
Our data is now ready to feed the network, let's see the structure 
of the network we're building today.


### Network structure
<img src="imagetwo.svg" alt="neural network python for regression" />


The network will receive a vector with four elements (nutrients), the first hidden
layer contains eight neurons, the second one four neurons and the output layer
will have just one single output (estimation of rice production). The
equations you see at each layer will be calculated by **Pytorch**, 
so you don't have to worry about.

### Network stages
However, you should know that building this kind of networks require three
steps, <em class="important-teal">Forward-Propagation</em> (calculations from left to right - equations in the
pic), <em class="important-orange">Loss-Function</em> (comparison between estimation and target) and
<em class="important-teal">Back-Propagation</em> (weights correction - requires 
gradients and <a href="/posts/gradient-descent-algorithm-in-python
"></a> algorithm).


With that being said, let's begin coding the network.


```python
# Neural Network (class)
class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.linear_one = nn.Linear(4, 8)
    self.linear_two = nn.Linear(8, 4)
    self.linear_three = nn.Linear(4, 1)
    self.selu = nn.SELU()
```

Let me explain what these lines mean, we're creating a class that will inherit
from the **nn.Module**, the **super** line will let us initialize our own variables,
next three lines are the layers with their respective neurons viewed from left
to right, and **SELU** is the activation function (similar to **RELU** but sometimes better).


Keep in mind that we're just declaring the tools needed to create the network, to
use them let's begin coding the forward-pro segment, which we'll add to the 
class **NeuralNet** (it's just a method of the class).

```python
  def forward(self, x):
    out = self.linear_one(x)
    out = self.selu(out)
    out = self.linear_two(out)
    out = self.selu(out)
    out = self.linear_three(out)
    return out
```

The forward method will expect a **x_instance** (nutrients 
combination) and return one output (the target estimation).
The network is already created but not trained yet, to do that 
we should start declaring the model, loss function and optimizer.


```python
# These lines of code does not belong to the class NeuralNet
model = NeuralNet() # creating an instance of the NeuralNet object
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 10_000
```

Criterion is **median-square-error** function, which is the one we're using as 
loss function, optimizer is the gradient-descent algorithm to optimize the
model and epochs is the number of times we loop throughout the whole dataset.



Mean Square Error - Loss Function:
<img src="img2.svg" alt="mse function - neural net - regression"/>
Well, let's train the model: 


### Training the network
```python
# Training the network
for epoch in range(epochs):
  x_split = X_train.split(16)
  y_split = y_train.split(16)
```

First we get into a loop splitting **x_train** and **y_train** into
mini-batches of $16$ instances ($16$ nutrient combinations),
we don't need to pass just a single combination to the
network as computers can deal even with thousands of instances
at once. However it's believed that computers work better
with batches whose length is multiple of $8$, like $8$, $16$, $32$, $64$, $128$ ...


```python
  for x_inst, y_inst in zip(x_split, y_split):
    y_hat = model.forward(x_inst)
    loss = criterion(y_hat, y_inst)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Now we get into an another loop (inside the first one),
we iterate throughout the object containing the mini-batches 
(**x_split**, **y_split**) calculating first
the estimation of target (forward-pro), then we compare
$\hat{y}$ (estimation) with $y$ (target) and then we get the loss function
value (a.k.a error).

Once we get the loss function value, it's time to call
**loss.backward** method, it calculates the corresponding
derivatives (gradients), then by using the **optimizer.step**
method we apply the **back-propagation** part, where the 
optimizer corrects weights (parameters), after that we
clean the gradients, that way during the next loop (mini-batch) 
gradients are tracked from scratch.

Lastly we want to validate the network as it learns, therefore 
using **torch.no_grad** we can isolate the network 
and thus gradients are not tracked. It will give as the chance to see 
how is the network doing with the dataset **x_valid** and **y_valid**
(a dataset our network never saw before).

To achieve it we just need to calculate the loss function using **y_valid**
and the estimation coming from the network for **x_valid** (a.k.a $\hat{y}$).

```python
  with torch.no_grad():
    y_hat_val = model(X_valid)
    loss = criterion(y_hat_val, y_valid)
    print(f"Epoch: {epoch} && Loss {loss}")
```

Keep in mind that we want to find a point where
the difference between the estimation ($\hat{y}$) and real ($y$) value
is minimum, thus a network that's truly learning should have
a lower loss value with every iteration.

By Running the code you should see something like this;

```python
# You will probably have different numbers, 
# don't worry, just make sure it's decreasing.
Epoch: 0 && Loss 1.2863106727600098
              .
Epoch: 9997 && Loss 0.0006022227462381124
Epoch: 9998 && Loss 0.0006020868313498795
Epoch: 9999 && Loss 0.000601950101554393
```

The model is already trained, let's see how well it is
doing with some combinations taken from the test 
dataset (our network never saw this data).


```python
# estimation for combinations [1, 18, 32, 68, 99] (X_test)
y_hat = model.forward(X_test[[1, 18, 32, 68, 99]])

# removing the scale done before
y_hat = y_scaler.inverse_transform(
    y_hat.detach().numpy())

# observed rice production
y_real = y_scaler.inverse_transform(
    y_test[[1, 18, 32, 68, 99]])
```

Now printing $y$ and it's estimation $\hat{y}$:


```sh
    y_real             y_hat
[412.31418614]  ==> [410.1303 ]
[387.02818856]  ==> [387.259  ]
[467.06371324]  ==> [467.66553]
[436.71754567]  ==> [434.823  ]
[230.06538564]  ==> [228.1431 ]
```

As you see the network is doing pretty well. This network can now be saved using **pickle** and be transferred to the research department of the example, they will feed the network with different combinations and it'll return a pretty accurate prediction.

In case you want to know how to deploy this network please read this <a href="/posts/deploy-neural-network-with-flask">post</a>.



