Today's challenge is the creation of a math model able to discriminate between
different species of penguins, to achieve it, we will make use of the
perceptron structure, which is a pretty simple model quite used to tackle
easy tasks.

Our dataset consists on describing two types of penguins by using the next
measurements:

- **Specie (Adelie: 1, Gentoo: -1) => Our target**
- Island the penguin was found at $(Torgersen: 1, Biscoe: 2, Dream: 3)$
- Bill Length and Depth (mm)
- Flipper Length (mm)
- Body Mass (grams)
- Sex $(female=0, male=1)$

As you may know penguins around the world share some physical aspects,
nevertheless, there're some attributes that make them different among
other penguins species. Can we create an AI model able to tell us what specie
a penguin belongs to, based on his physical characteristics? Well that's
precisely what we're creating today.

Since perceptron capacities are limited and the purpose of this tutorial, our
model will only distinguish between two species, Gentoo and Adelie (the
original dataset has three species as you may see in the picture). Let's give
a look to our data (random sample):

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

To build our model you should be familiar with Python classes and the library
Numpy. The dataset we're working with can be found [here][2]; in case you want to
know the transformations were made to the original dataset go to this [link][2].

[1]: https://files.wmatrix.xyz/penguins/penguins.csv
[2]: https://files.wmatrix.xyz/penguins/transformations.json

To get more info about the original paper and dataset publisher, please visit this [place][3]. With that being said, let's do it.

The perceptron structure:

<img src="imageone.png" alt="perceptron python neural network" />

Building the model will be done by following the next stages, please don't
miss them during the process.

- Forward propagation.
- Loss function value calculation.
- Weights correction.
- Repeat the process.

First of all let's start importing the libraries Numpy and Pandas. Numpy will
help us to deal with the numerical calculations and Pandas just contains a
pretty useful function to read csv files (a dataframe in our case). We also
need a data splitter and scaler which we'll import from the Sklearn library.

Next lines do import the libraries and load the dataset.

```py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('./penguins.csv')
```

To proceed, we need to get our target $(specie = y)$ in a vector and the features in an array rest of features in the dataset $X$, then
creating $y$ and $X$ would be like.

```py
y = dataset.species.to_numpy()
X = dataset.iloc[:, 1:].to_numpy()
```

Loaded the data, it's needed to create three different samples, one for
training, other for validation and another one to test. 90% of data will be
dedicated to train and validate the model, the rest for testing. By the way,
our model wont never see the testing sample during its construction.

<img src="imagetwo.png" alt="split dataset python - data science" />

To get it done, let's use the train_test_split function from Sklearn.

```py
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size = 0.1, random_state=10)

X_train, X_validation, y_train, y_validation = train_test_split(
X_train, y_train, test_size = 0.15, random_state=10)
```

Now our data is almost ready, it's clean, splitted and well organized. In
order to avoid not desirable consequences, it's necessary to scale it using
the StandardScaler function from Sklearn. To be honest due to the simplicity
of perceptrons it might be not needed, however it's just a good practice.
Please keep in mind that the scaler parameters will come from the training
dataset.

```py
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)
```

So far we've seen how to deal with the dataset, it's time to start coding the
perceptron. By using a Python class, we will develop the construction piece by
piece; During this process please follow the stages showed before.

The first step is about defining the class and the parameters it will expect to be created.


```py
class Perceptron:
    def __init__(self, X_matrix, y_matrix, epochs=1, n_features=11, batch_size=2):
        self.X = X_matrix
        self.y = y_matrix
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = np.random.randn(n_features)
        self.error = np.random.randn()

```
Let me explain what these variables mean.

$X\_matrix$ and $y\_matrix$ are our training data;
$Epochs$ is the number of times we loop
throughout the whole dataset (one by default - perceptrons make progress so
fast), $n\_features$ is the number of features in
our dataset (six in our case), $batch\_size$ is the
number of instances we're taking per iteration.

The weights and bias get random values as initial value, but with every
iteration they will be modified (corrected).

Forward propagation take a form like this:

```py
    def propagation(self, weights, x_matrix):
        y_hat = np.matmul(weights, x_matrix.transpose()) + self.error
        return y_hat
```

The loss function: its objective will be the comparison between the output we
got from the forward propagation $y\_hat$ and the real target $y$, when both
get closer (the only one changing is y_hat) the loss function value will get
smaller.

The perceptron is usually created with the Hinge loss function. If you want to
understand this function and its derivatives (needed to correct the weights),
you should see [this][4].

[4]: <https://files.wmatrix.xyz/penguins/transformations.json> ""

Please remind that since we're working with vectors, taking means is needed to
get a representative and unique value. It means we don't want to know how well
is the model doing per instance, instead, the well it's doing generally.

Applying the loss function in our class.

```py
    def loss_function(self, y_true, y_hat):
        boundary = y_true * y_hat
        boundary = np.where(boundary > 1, 0, 1 - boundary)
        return boundary.mean()
```

Once we've a loss function, this function derivatives are needed to correct
the weights at each iteration (the optimizer we're using is the [gradient-descent][5].

[5]: <https://www.deepmatrix.xyz/posts/gradient-descent-algorithm-in-python> ""


```py
  def derivatives(self, y_true, y_hat, x_matrix):
      boundary = y_true * y_hat
      reference = np.where(boundary > 1, 0, -y_true)
      weights_dvt = (np.expand_dims(reference, 1) * x_matrix).mean(0)
      errors_dvt = reference.mean()
      return weights_dvt, errors_dvt
```

Now creating a simple function for correcting the weights:

```py
  def weights_correction(self, weights_dvt, errors_dvt):
      self.weights -= weights_dvt
      self.error -= errors_dvt
```

Our Perceptron is done. Lastly we'll create a function to train it

```py
  def train(self):
      for epoch in range(self.epochs):
          for index in np.arange(0, len(self.y), self.batch_size):
              x_instance, y_instance = (self.X[index: index + self.batch_size], \
                                       self.y[index: index + self.batch_size])
              y_hat = self.propagation(self.weights, x_instance)
              loss_value = self.loss_function(y_instance, y_hat)
              weights_dvt, errors_dvt = self.derivatives(y_instance, y_hat, \
                                                         x_instance)
              self.weights_correction(weights_dvt, errors_dvt)

              # printing validation values
              y_hat_validation = self.propagation(self.weights, X_validation)
              loss_value_validation = self.loss_function(y_validation, y_hat_validation)
              print(f'epoch {epoch} and batch {index}-{index + self.batch_size} ===> {loss_value_validation}')

```

As you may see we're looping just once throughout X_train (epochs: 1),
then it gets into another loop following the next structure:

- Take 2 instances per loop (batch_size: 2)
- Calculate y_hat and loss_value
- Get the current derivatives and correct the weights
- Repeat the process (X_train.length / batch_size) times.

We want to validate our model at each iteration and see how good is doing on
the sample X_validation (give a look to the line #printing validation
values).

Well the code is done. It will take you few minutes to figure out how it works,
however, a technique you can apply to understand it, is the well known reverse
engineering, will help you a lot.


Let's train our model:

```py
model = Perceptron(X_train, y_train, n_features=6, batch_size=2)
model.train()
```

You should see something like this (not the same figures, remind weights
initiated randomly):

```py
epoch 0 and batch 0-2 ===> 0.4864170568486674
epoch 0 and batch 2-4 ===> 0.3569053227079397
epoch 0 and batch 4-6 ===> 0.3569053227079397
epoch 0 and batch 6-8 ===> 0.02141807371487539
epoch 0 and batch 8-10 ===> 0.02141807371487539
epoch 0 and batch 10-12 ===> 0.02141807371487539
epoch 0 and batch 12-14 ===> 0.0
epoch 0 and batch 14-16 ===> 0.0
```

Now our Perceptron is able to know what specie a penguin belongs to; To prove
it, we can use the testing sample (the model never saw it before):

```py
y_hat = model.propagation(model.weights, X_test)
y_hat = np.where(y_hat > 0, 1, -1)
(y_hat == y_test).mean()
```

A mean equal to 1.0 means our model did forecast all instances in the testing
sample rightly, a value of 0.5 or lower means it's doing pretty bad and
something in the process went wrong.

```py
1.0
```

We can confirm the model is able to recognize a penguin just by his
characteristics. This kind of things can be achieved by artificial
intelligence easily, today we used the simplest model, imagine what we can do using
more complex things.

I hope you enjoyed building this AI, there are some related posts below in
case you want to continue learning.

