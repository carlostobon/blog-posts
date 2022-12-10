One of the most common optimizers in the artificial
intelligence field is the quite famous gradient
descent algorithm. Sometimes it may be hard to understand
the role it plays to make an AI model works, in order to get a better
picture of this process, we'll optimize a
function using **Numpy** and some basic calculus in **Python**.

By optimizing I mean doing something pretty similar to what we
do when we're building neural networks or just using a regular
machine learning model.

## The Rosenbrock function

The function we're using today is the well known
Rosenbrock function, it has some particularities
that make it perfect to test an optimization algorithm
like gradient-descent, for Instance, this function
contains a <em class="important-orange">global minimum</em> 
inside a parabolic shaped
flat valley and finding it is not an easy task.

We can describe the Rosenbrock function as:

<div class="math-div">

$$
f(x, y)=(a-x)^2 + b(y - x^2)^2\ with\ a=1,\ b=100
$$

</div>

## Gradient-descent

The gradient-descent algorithm is an 
<em class="important-teal">iterative method</em> to
find critic points of a function, it could be saddle,
minimum or maximum points.

So summarizing, Our goal is to find the critic points
of the Rosenbrock function by using the gradient-descent
algorithm.

### Gradient structure:

A function's gradient is just a vector that contains all its
derivatives, of course this is an extremely simple way to describe
such a beautiful expression but should be enough to understand it.

The gradient can be described as:

<div class="math-div">

$$
\nabla f(x,y)= \frac{\partial}{\partial x}f(x,y), \frac{ \partial}{ \partial y} f(x,y)
$$

</div>

### Gradient descent algorithm:

Please keep in mind that this method is an iterative one,
so to make it works we'll use loops and a random point
as starting point. The algorithm may be expressed as:

<div class="math-div">

$$
X_{k+1} = X_{k} - \alpha \nabla f(x,y)
$$

</div>

### Rosenbrock function derivatives:

In order to build the function's gradient we must
first get its <em class="important-orange">derivatives</em>, 
which can be written as:

With respect to $x$:

<div class="math-div">

$$
\frac{\partial}{\partial x}f(x,y) = -2(a-x) - 2^2bx(y-x^2)
$$

</div>

With respect to $y$:

<div class="math-div">

$$
\frac{\partial}{\partial y}f(x,y) = 2bx(y-x^2)
$$

</div>

## The Derivative

In case you forgot what a derivative is, it's just a mathematical expression
that tell us how a function changes once one of its variables change given a
specif point. This is not a rigorous definition but probably enough to accomplish our
goal.

## How gradient-descent works?

We already know what the gradient means, however if you see again the
gradient-descent expression, you will see $\alpha$ multiplying the gradient,
$\alpha$ is a variable setting the speed we use to make progress on getting
the critic points, $\alpha$ should be set between $0$ and $1$ to guarantee the
algorithm works.

We're in charge to give a value to $\alpha$, an ideal $\alpha$ must be a one
that take us pretty close to the real <em class="important-orange">critic point</em>, many papers have been
written about getting the right $\alpha$, however it will be up to us to decide
what value is giving us the best results.

When you're dealing with a complex model you may be forced to use a small $\alpha$,
at least one really close to zero; working with small alphas will lead to a
high computational cost, however, regular computers are powerful enough
to tackle this kind of tasks.

### Gradient-descent step by step

As we saw before the algorithm is iterative, a friendly way to
understand how it works would be:

1. Initiate with at a random point $(x, y)$.
2. Calculate the gradient at that point.
3. Multiply the gradient by $\alpha$
4. Correct the point by subtracting the output gotten in literal three.
5. Repeat the process starting from literal two.


## Building the algorithm

So assuming everything is clear so far, let's start coding it. Remember
we'll be using **Python**, specifically its numerical library **Numpy**

<div class="special-ivory">
    Numpy is perfect to work with vectors and matrices, also it was
    written using a lower level programming language.
</div>

Well, let's begin importing **Numpy** and defining the Rosenbrock function.

```python
import numpy as np

def rosenbrock(point: tuple , a=1, b=100) -> float:
  x, y = point
  return (a - x) ** 2 + b * (y - x**2) ** 2
```

Evaluating the function at a random point like:

```python
print("f(1, 2) = ", rosenbrock((1, 2)))
print("f(3,-2) = ", rosenbrock((3,-2)))
```

```sh
f(1, 2) =  100
f(3,-2) =  12104
```

We said the <em class="important-orange">gradient</em> 
is a vector with the respective derivatives,
so coding it from the formulas shown before and putting them in a
vector:

```python
def gradient(point: tuple, a=1, b=100) -> np.array:
    x, y = point
    x_dev = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    y_dev = 2 * b * (y - x **2)
    return np.array((x_dev, y_dev))
```

Calculating the gradient at a random point:

```python
print("grad at: (0.8, 1.2) = ", gradient((.8, 1.2)))
print("grad at: (3.5,-0.2) = ", gradient((3.5,-0.2)))
```

```sh
grad at: (0.8, 1.2) =  [-179.6  112. ]
grad at: (3.5,-0.2) =  [17435. -2490.]
```

Both things seems to be working, let's code the algorithm. We'll
create a **Python** <em class="important-teal">class</em> which is a 
structure that allow us to define properties and methods.

```python
class GradientDescent:
    def __init__(self, alpha=0.001, dimensions = 2, iterations = 10_000):
      self.location = np.random.rand(dimensions)
      self.alpha = alpha
      self.iterations = iterations
```

Once we declare an object using this class, such object will have by default
three properties, $\alpha$ which was explained before, **dimensions** which is needed
to generate the initial random point, by the way it's equal to two cause our function
has two dimensions $f(x,y)$, and lastly **iteration**, just a tentative number of times to iterate the algorithm, it should be enough to reach the global minimum.

Now we need to add a method to the class in order to run the algorithm.

```python
    def run(self):
      for i in range(self.iterations):
        print('iter', i, ': f(x, y) =', rosenbrock(self.location), '->', self.location)
        self.location -= self.alpha * gradient(self.location)
```

This last piece of code is literally applying the 
<em class="important-teal">five steps</em> needed to make
the algorithm works; we'll get into a loop that will last $10k$ iterations, on
every iteration we calculate gradients and the function value, that way it's 
possible to correct the location, that location will be the critic point we've 
been looking for.

## Running the gradient-descent algorithm
Creating an instance:

```python
model = GradientDescent()
```

Let's give a look to the initial values:

```python
print(model.location)
print(model.alpha)
print(model.iterations)
```

```sh
loaction: [0.1820554  0.76995654]
alpha: 0.001
iterations: 10000
```

Keep in mind that your initial location is different to mine, 
but final location *(global minimum)* should be quite similar.

Running the model:
```python
model.run()
```

Last four iterations:

```sh
iter 9996 : f(x, y) = 0.00004144 -> [0.997965 0.995927]
iter 9997 : f(x, y) = 0.00004141 -> [0.997966 0.995929]
iter 9998 : f(x, y) = 0.00004138 -> [0.997967 0.995930]
iter 9999 : f(x, y) = 0.00004134 -> [0.997968 0.995932]
```

As we said before, this function has a global minimum at $(x=1, y=1)$ where the
function takes a value of $z=0$; What we did today is just a similar thing 
of what we usually do when creating AI models, it wouldn't be the Rosenbrock
function, but a loss function depending on the problem we're tackling.

I hope you enjoyed this post. If you like playing with code and data, 
there're some other posts here you may find interesting.

