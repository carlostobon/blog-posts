Throughout this post we'll be creating an artificial intelligence model
able to discriminate between two different species of penguins, to achieve it, 
we're going to use the perceptron structure, which is a
simple model quite useful to tackle this type of tasks.


Penguins around the world share some physical aspects,
however, there're some attributes that make them different among
other penguin species. Things like the place they live at or the
size of their bodies are key to distinguish between different kind
of this animals.

We've a dataset that give us information about penguins
that were observed by scientists during a certain period of time,
they took some measurements creating a dataset that looks like:


- Specie (**Adelie**: $1$, **Gentoo**: $-1$) $\rightarrow$ Our target
- Island the penguin was found at (Torgersen: $1$, Biscoe: $2$, Dream: $3$)
- Bill Length and Depth (mm)
- Flipper Length (mm)
- Body Mass (grams)
- Sex (female: $0$, male: $1$)

## The challenge
Given this data, can we create a simple AI model with the capacity to tell us 
<em class="important-orange">what specie a penguin belongs to</em> based on the other six characteristics? 
That's precisely what we're creating today, a model that we feed with those six
measurements and it returns the type of penguin. 

### The dataset
Let's give a look to our data (random sample):

<div class="table">
  <table>
    <thead>
      <tr>
        <th>index</th>
        <th>species</th>
        <th>island</th>
        <th>bill_length</th>
        <th>bill_depth</th>
        <th>flipper_length</th>
        <th>body_mass</th>
        <th>sex</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>244</th>
        <td>-1</td>
        <td>2</td>
        <td>49.5</td>
        <td>16.1</td>
        <td>224</td>
        <td>5650</td>
        <td>1</td>
      </tr>
      <tr>
        <th>161</th>
        <td>-1</td>
        <td>2</td>
        <td>40.9</td>
        <td>13.7</td>
        <td>214</td>
        <td>4650</td>
        <td>0</td>
      </tr>
      <tr>
        <th>110</th>
        <td>1</td>
        <td>2</td>
        <td>45.6</td>
        <td>20.3</td>
        <td>191</td>
        <td>4600</td>
        <td>1</td>
      </tr>
      <tr>
        <th>11</th>
        <td>1</td>
        <td>1</td>
        <td>41.1</td>
        <td>17.6</td>
        <td>182</td>
        <td>3200</td>
        <td>0</td>
      </tr>
      <tr>
        <th>40</th>
        <td>1</td>
        <td>3</td>
        <td>40.8</td>
        <td>18.4</td>
        <td>195</td>
        <td>3900</td>
        <td>1</td>
      </tr>
      <tr>
        <th>173</th>
        <td>-1</td>
        <td>2</td>
        <td>46.5</td>
        <td>14.5</td>
        <td>213</td>
        <td>4400</td>
        <td>0</td>
      </tr>
      <tr>
        <th>243</th>
        <td>-1</td>
        <td>2</td>
        <td>45.5</td>
        <td>14.5</td>
        <td>212</td>
        <td>4750</td>
        <td>0</td>
      </tr>
      <tr>
        <th>36</th>
        <td>1</td>
        <td>3</td>
        <td>42.2</td>
        <td>18.5</td>
        <td>180</td>
        <td>3550</td>
        <td>0</td>
      </tr>
    </tbody>
  </table>
</div>

### Requirements
To build this model you should be familiar with **Python** classes and the library
**Numpy**. The dataset we're working with can be found [here][1], and the original paper and dataset [here][2].

[1]: https://files.wmatrix.xyz/penguins/penguins.csv
[2]: https://files.wmatrix.xyz/penguins/transformations.json


## The perceptron structure:
<img src="imageone.svg" alt="perceptron python neural network" />

Building the model will take the next four stages, please don't
miss them during the process.

- Forward propagation.
- Loss function value calculation.
- Weights correction.
- Repeat the process.

## Coding the perceptron
First of all let's start importing the libraries **Numpy** and **Pandas**. **Numpy** will help us to deal with the numerical calculations and **Pandas** just contains a
pretty useful function to read csv files (a dataframe in our case). We also
need a data splitter and scaler which we'll import from the **Sklearn** library.

### Importing libraries
The following lines import the libraries and load the dataset.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('./penguins.csv')
```

To proceed, we need to get our **target** |<em class="important-orange">specie</em> = $y$| in a vector and the **features** in an array |<em class="important-orange">
rest of features</em> = $X$|, so creating $y$ and $X$ would be like.


```python
y = dataset.species.to_numpy()
X = dataset.iloc[:, 1:].to_numpy()
```

### Splitting the data
Loaded the data, it's needed to create two different samples, one for
<em class="important-orange">training</em> and another one for 
<em class="important-orange">testing</em>. $90\%$ of the data will be
used to train the model, the rest for testing. Keep in mind
that our model wont never see the testing sample during its construction.

<img src="imagetwo.svg" alt="split dataset python - data science" />

To get the samples we can use the **train_test_split** function from **Sklearn**.

```python
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.1, random_state=10)
```

Setting the data is almost done, it's clean, split and well organized.
The next step is to scale it using the **StandardScaler** function from Sklearn. 
Being honest, due to the simplicity of perceptrons it might be not needed, 
however, it's a good practice.
Please keep in mind that the scaler parameters will come from the training
dataset.

### Scaling the data

let's create an scaler based on the normal distribution standarization:


```python
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)
```


The data is ready to feed the model, now we can continue coding the
perceptron. By using Python classes, we'll develop the construction piece by
piece. During this process please follow the stages cited before.

### Defining a perceptron class
The first step is the definition of the class and the parameters it will 
expect to be created.

```python
class Perceptron:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.randn(6, 1)
        self.error = np.random.randn(1,1)
```

**X_train** and **y_train** are our training data,
**weights** initiate at a random point, same thing with **error**,
by the way, the number of rows in **weights** should be equal
to the number of features in our dataset; **Weights** and **error**
will be evaluated and corrected at each iteration

### Forward Propagation
Well now it's time to add the <em class="important-teal">
forward-propagation</em> method to the class.

<div class="math-div">

$$
\hat{y}(w, b) = xw + b
$$

</div>

<div class="special-ivory">
Forward propagation works from left to right in the perceptron structure
and give us an estimation for the target, such estimation is called y_hat.
</div>


```python
  def forward(self):
    self.y_hat = np.matmul(self.X_train, self.weights) + self.error
```

### Loss function
The loss function objective will be the comparison between the output we
get from the **forward propagation** $\hat{y}$ and the real target $y$, when both
get closer (<em>the only one changing is $\hat{y}$</em>) the loss function value will get smaller. Our goal is to get the loss function as small as we can.

Perceptrons are usually built using the Hinge loss function:

<div class="math-div">

$$
L(y, \hat{y}) = 
\begin{cases}
  1 - y_i \hat{y_i} & \text{if } y_i \hat{y_i} <  1\\    
  0 & \text{otherwise}
\end{cases}
$$

</div>

Can also be defined as:

<div class="math-div">

$$
L(y, w, b) = 
\begin{cases}
  1 - y_i (xw + b) & \text{if } y_i \hat{y_i} <  1\\    
  0 & \text{otherwise}
\end{cases}
$$

</div>

Since we're working with matrices, we got to use an **average** as a 
unique and representative value. It means we don't want to know how well
is the model doing per instance (*row*), instead, the well it's doing generally.

Adding the loss function to our class:

```python
  def loss_value(self):
    self.loss = np.maximum(0, 1 - self.y_train * self.y_hat).mean(0)
```

Once we've the loss function set, this function derivatives are required to correct
the weights at each iteration. Such derivatives can be written as:

<div class="math-div">

$$
\frac{\partial L}{\partial w} = 
\begin{cases}
  -y_i x & \text{if } y_i \hat{y_i} < 1\\
  0 & \text{otherwise}
\end{cases}
$$

</div>


<div class="math-div">

$$
\frac{\partial L}{\partial b} = 
\begin{cases}
  -y_i & \text{if } y_i \hat{y_i} < 1\\
  0 & \text{otherwise}
\end{cases}
$$

</div>

Adding them to our class:

```python
  def back(self):
    boundary = self.y_train * self.y_hat
    reference = np.where(boundary < 1, -self.y_train, 0)
    self.weights_der = (reference * self.X_train).mean(0)
    self.error_der = reference.mean(0)

    self.weights_der = np.expand_dims(self.weights_der, 1)
    self.error_der = np.expand_dims(self.error_der, 1)
```

<div class="special-ivory">
  back propagation works from right to left in the perceptron structure,
  it gets the correspondent derivatives in a given point.
</div>


Now creating a simple method to correct the weights, 


```python
  def weights_correction(self, alpha = 1 / 1000):
    self.weights -= alpha * self.weights_der
    self.error -= alpha * self.error_der
```

<div class="special-ivory">
  The optimizer we're using is the gradient-descent.
</div>

Our **perceptron** is built. Let's create an instance of it.

```python
model = Perceptron(X_train, y_train)
```

This object <em class="important-orange">model</em> contains
all parameters and methods we defined in our class. Now training
the model would be like:


```python
for i in range(10000):
  model.forward()
  model.loss_value()
  model.back()
  model.weights_correction()
  print("iter:", i, "loss value:", model.loss)
```

You should see something like this with different values, but make
sure it's decreasing with every iteration.

```sh
iter: 9997 loss value: [0.00097652]
iter: 9998 loss value: [0.00097646]
iter: 9999 loss value: [0.00097641]
```

When you're creating a model for production, you shouldn't use a training
set to get a loss value, instead, it's better to create another sample like
a validation set, however, this goes beyond this post.

Once the model is trained we can create a new instance of the Perceptron class
to test the results with the testing set. To achieve it, we just need to
transfer the weights and error from the trained model to the new one.


```python
model_test = Perceptron(X_test, y_test)
model_test.weights = model.weights
model_test.error = model.error
```


Now testing the model:

```python
model_test.forward()
y_test_hat = np.where(model_test.y_hat > 0, 1, -1)
print("accuracy: "(y_test_hat == y_test).mean())
```

```sh
accuaracy: 1.0
```

Recall our model never saw the testing sample in its training, however,
it's really good forecasting the type of penguin once is feeding with the
six features. 

We can confirm the model is able to recognize a penguin just by his
characteristics. This kind of things can be achieved easily using artificial
intelligence, the Perceptron is probably the most basic model and see the things
we can do, impressive.

I hope you enjoyed building this AI, there are some related posts around in
case you want to continue learning.

