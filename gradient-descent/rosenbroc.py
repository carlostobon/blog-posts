import numpy as np

def rosenbrock(point: tuple , a=1, b=100) -> float:
  x, y = point
  return (a - x) ** 2 + b * (y - x**2) ** 2


print("f(1, 2) = ", rosenbrock((1, 2)))
print("f(3,-2) = ", rosenbrock((3,-2)))


def gradient(point: tuple, a=1, b=100) -> np.array:
    x, y = point
    x_dev = -2 * (a - x) - 4 * b * x * (y - x ** 2)
    y_dev = 2 * b * (y - x **2)
    return np.array((x_dev, y_dev))

print("grad at: (0.8, 1.2) = ", gradient((0.8, 1.2)))
print("grad at: (3.5,-0.2) = ", gradient((3.5,-0.2)))


class GradientDescent:
    def __init__(self, alpha=0.001, dimensions = 2, iterations = 10_000):
        self.location = np.random.rand(dimensions)
        self.alpha = alpha
        self.iterations = iterations

    def run(self):
        for i in range(self.iterations):
            print('iter', i, ': f(x, y) =', rosenbrock(self.location), '->', self.location)
            self.location -= self.alpha * gradient(self.location)

model = GradientDescent()

print('loaction:', model.location)
print('alpha:', model.alpha)
print('iterations:', model.iterations)

gradient_descent.run()

