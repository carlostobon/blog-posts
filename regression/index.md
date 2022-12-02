
When you're getting into the deep-learning field, it's quite important to understand how
neural networks work, many of the mathematical models that you'll find in AI
are based in neural networks, therefore, learning how to code one could be
crucial to jump into any job related with this field.


Most time you will find tutorials explaining how to make a neural network for
classification problems (cat and dog), however, when you're a scientist or
data analyst, it's much more common to find yourself needing to tackle a
regression kind problem.


Let's suppose that you work for the **Company-XYZ**, this company is in the business
of growing rice, as many companies out there, getting new ways to make more money is 
key to survive in the market, thus, leaders in the company believe that finding 
a way to grow rice in a faster way could be a great option (they're probably right).


The research department starts testing with four new type of nutrients to
improve the growing conditions creating in the way a dataset with $1000$ 
instances that looks like:


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

You're just seeing the first four experiments but keep in mind that we've a
thousand of them in the dataset, the first row (experiment) means that a
combination of four nutrients ($12$, $16$, $8$, $2$) led to produce $100$ tons of rice,
same thing for all following rows.

As we don't have a real dataset, we got to create one to simulate this
scenario, let's begin importing some libraries to create our dataset.

```py
# Libraries
from sklearn.datasets import make_regression

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
two objects, <em class="important-orange">features</em> is catching a matrix with four columns that will
represent the nutrients and <em class="important-orange">target</em>the total production.

**n_samples** means we want $1000$ instances, **n_features** the number of columns it'll have (nutrients), **noise** is just a standard deviation we apply to the target (makes the simulation more real) and **random_state** is the seed, pseudo-random numbers need a seed to be generated.

Lastly we want to add $10$ units to features and $400$ to target, that way we can
achieve a dataset pretty similar to the one explained in the example (rice
production). Let's give a look to our data.

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


It's just a couple of rows but if you use the same random_seed used in this
tutorial, those values should be identical to yours; **features** should be $(1000, 4)$ 
and **target** $(1000,)$. Only thing left is reshaping the target vector, 
we want it to be $(1000, 1)$.

```py
target = target.reshape(-1, 1)
```


The next step is splitting the dataset, it means taking something like $80\%$ of
data to train the network, and the rest $20\%$ for validation and testing.


<img src="imageone.png" alt="dataset split" />


To split the dataset we're gonna use the Sklearn's splitter, so let's import
it and split the data.


```py
# Libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
```

```py
# Splitting dataset
X_train, X_valid, y_train, y_valid = train_test_split(
  features, target, test_size=0.2, random_state=RANDOM_STATE)

X_valid, X_test, y_valid, y_test = train_test_split(
  X_valid, y_valid, test_size=0.5, random_state=RANDOM_STATE)
```

Now we've three datasets, it's time to scale them using the mean and standard
deviation taken from <em class="important-teal">X_train</em> and <em class="important-teal">y_train</em> 
as shown in the next figure.

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


As we're going to use <em class="important-teal">Pytorch</em> to create the network,
we must transform our arrays to tensors (**Pytorch** objects), it will
let us to work with **Pytorch** and track the derivatives (gradients) 
throughout the calculations, it's needed to do the back-propagation 
section (we'll see more about it later).

<div class="special-ivory">
Pytorh is a Python library to create AI
</div>


To achieve both things (split and transform) we're going 
to create a function called **transformer**, which
will receive our datasets and return them scaled and 
converted in tensors. So, let's
start adding some functions to the importing section.


```py
# Libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.functional as F
```

Creating the scalers:

```py
# scaler (it gets and holds the mean & std of X_train and y_train)
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)
```

And the function **transformer**:

```py
# Transformer function
def transformer(inputs, scaler) -> list:
  to_return = list()
  for item in inputs:
    scaled = scaler.transform(item)
    scaled = torch.tensor(scaled, dtype=torch.float)
    to_return.append(scaled)
    return to_return
```


This function is expecting a scaler object which contains the mean and
std $(\mu, \sigma)$ of **X_train** or **y_train**, also a list with the items to be scaled,
lastly is returning a list with tensors already scaled.


Now using the **transformer** function:
```py
# Transforming the datasets
X_train, X_test, X_valid = transformer(
  [X_train, X_test, X_valid], x_scaler)

y_train, y_test, y_valid = transformer(
  [y_train, y_test, y_valid], y_scaler)
```

We can say data is now ready to feed the network, let's see the structure 
of the network we're building today.


<img src="imagetwo.png" alt="neural network python" />


The network will receive a vector with four elements (nutrients), the first hidden
layer contains eight neurons, the second one four neurons and the output layer
will have just one single output (estimation of rice production). The
equations you see at each layer will be calculated by **Pytorch**, 
so you don't have to worry about.

However, you should know that building this kind of networks require three
steps, <em class="important-teal">Forward-Propagation</em> (calculations from left to right - equations in the
pic), <em class="important-orange">Loss-Function</em> (comparison between estimation and target) and
<em class="important-teal">Back-Propagation</em> (weights correction - requires 
gradients and [gradient-descent][6] algorithm).

[6]: <https://www.deepmatrix.xyz/posts/gradient-descent-algorithm-in-python> ""

With that being said, let's begin coding the network.


```py
# Neural Network (class)
  class NeuralNet(nn.Module):
    def __init__(self):
      super(NeuralNet, self).__init__()
      self.linear_one = nn.Linear(4, 8)
      self.linear_two = nn.Linear(8, 4)
      self.linear_three = nn.Linear(4, 1)
      self.selu = nn.SELU()
```

Let me explain what those lines mean; We're creating a class that will inherit
from the **nn.Module**, the **super** line will let us initialize our own variables,
next three lines are the layers with their respective neurons viewed from left
to right and **SELU** (similar to **RELU** but sometimes better) is the activation function.


Keep in mind that we're just declaring the tools needed to create the network, to
use them let's begin coding the forward-pro segment, which we'll add to the 
class **NeuralNet** (it's just a method of this class).

```py
  def forward(self, x):
    out = self.linear_one(x)
    out = self.selu(out)
    out = self.linear_two(out)
    out = self.selu(out)
    out = self.linear_three(out)
    return out
```

The forward method will expect a **x_instance** (nutrients 
combination) and return one output (the target estimation), 
the network is already created but not trained yet, to do that 
we should start declaring the model, loss function and optimizer.


```py
# These lines of code does not belong to the class NeuralNet
model = NeuralNet() # creating an instance of the NeuralNet object
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 10_000
```
These lines are self explanatory, we're using the stochastic [gradient-descent][7] 
algorithm to find the minimum of the **median-square-error** function (our loss function),
this function has a minimum where the prediction 
(estimation) is equal to the target, like you can see in this pic. 

By the way, **epochs** is the number of times we loop throughout the whole
dataset (**X_train**, **y_train**).

[7]: <https://deepmatrix.xyz/posts/gradient-descent-algorithm-in-python> ""

Mean Square Error - Loss Function:

<img src="img2.svg" alt="mse function - neural net - regression"/>

Well, let's train the model: 


```py
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


```py
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
$\hat{y}$ (estimation) with $y$ (target) and get the loss function
value (a.k.a error).

Once we get the loss function value, it's time to call
**loss.backward** method, it calculates the corresponding
derivatives (gradients); Then by using the **optimizer.step**
method we apply the **back-propagation** part, where the 
optimizer corrects weights (parameters), after that we
clean the gradients, that way during the next loop (mini-batch) 
we track the gradients from scratch.

Lastly we want to validate the network as it learns, therefore 
using **torch.no_grad** we can isolate the network 
and thus gradients are not tracked. It will give as the chance to see 
how is the network doing with the dataset **x_valid** and **y_valid**
(a dataset our network never saw before).

To achieve it we just need to calculate the loss function using **y_valid**
and the estimation coming from the network for **x_valid** (a.k.a $\hat{y}$).

```py
  with torch.no_grad():
    y_hat_val = model(X_valid)
    loss = criterion(y_hat_val, y_valid)
    print(f"Epoch: {epoch} && Loss {loss}")
```

By Running the code you should see something like this;
Keep in mind that we want to find a point where
the difference between the estimation ($\hat{y}$) and real ($y$) value
is minimum, then a network that's truly learning should have
a lower loss value with every iteration.


```py
  # You probably have different numbers, 
  # don't worry, just make sure it's decreasing.
  Epoch: 0 && Loss 1.2863106727600098
  Epoch: 1 && Loss 1.1440201997756958
  Epoch: 2 && Loss 1.0215907096862793
                .
                .
                .
  Epoch: 9997 && Loss 0.0006022227462381124
  Epoch: 9998 && Loss 0.0006020868313498795
  Epoch: 9999 && Loss 0.000601950101554393
```

The model is already trained, let's see how well is it 
doing with some combinations taken from the test 
dataset (our network never saw this data).


```py
# estimation for combinations [1, 18, 32, 68, 99] (X_test)
y_hat = model.forward(X_test[[1, 18, 32, 68, 99]])

# removing the scalation done before
y_hat = y_scaler.inverse_transform(
    y_hat.detach().numpy())

# observed rice production
y_real = y_scaler.inverse_transform(
    y_test[[1, 18, 32, 68, 99]])
```

Now printing $y$ and it's estimation ($\hat{y}$):


```py
   y_real       ==>    y_hat
[412.31418614]  ==> [410.1303 ]
[387.02818856]  ==> [387.259  ]
[467.06371324]  ==> [467.66553]
[436.71754567]  ==> [434.823  ]
[230.06538564]  ==> [228.1431 ]
```


As you see the network is doing pretty well, however to
see more points it's common to plot the real and 
predicted value, a almost perfect prediction
would be a line with slope equal to one.


**Y_hat** vs **Y_real**:

<img src="img3.svg"/>

  
Please don't forget all these tests have been done with
the **x_test** and **y_test** dataset. To use this network you must 
first scale the input data with **x_scaler** and transform it to tensors, 
then once you got the output given by the network, remove the
scalation using the **inverse_transform** method from **y_scaler**.

Well that's all, I hope you've found this tutorial useful, 
to see how to create an API with this model go to this link.


