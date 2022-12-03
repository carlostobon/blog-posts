import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('./penguins.csv')

X = dataset.iloc[:, 1:].to_numpy()
y = dataset.species.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state=10)

# X_train, X_validation, y_train, y_validation = train_test_split(
    # X_train, y_train, test_size = 0.15, random_state=10)

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
# X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

y_train = np.expand_dims(y_train, 1)
y_test = np.expand_dims(y_test, 1)

class Perceptron:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = np.random.randn(6, 1)
        self.error = np.random.randn(1,1)

    def forward(self):
        self.y_hat = np.matmul(self.X_train, self.weights) + self.error

    def loss_value(self):
        self.loss = np.maximum(0, 1 - self.y_train * self.y_hat).mean(0)

    def back(self):
        boundary = self.y_train * self.y_hat
        reference = np.where(boundary < 1, -self.y_train, 0)
        self.weights_der = (reference * self.X_train).mean(0)
        self.error_der = reference.mean(0)

        self.weights_der = np.expand_dims(self.weights_der, 1)
        self.error_der = np.expand_dims(self.error_der, 1)

    # Weights correction
    def weights_correction(self, alpha = 1 / 1000):
        self.weights -= alpha * self.weights_der
        self.error -= alpha * self.error_der

model = Perceptron(X_train, y_train)

for i in range(10000):
    model.forward()
    model.loss_value()
    model.back()
    model.weights_correction()
    print("iter:", i, "loss value:", model.loss)





model_test = Perceptron(X_test, y_test)
model_test.weights = model.weights
model_test.error = model.error

model_test.forward()
y_test_hat = np.where(model_test.y_hat > 0, 1, -1)
print((y_test_hat == y_test).mean())




