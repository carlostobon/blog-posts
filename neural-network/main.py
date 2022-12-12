from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.functional as F



RANDOM_STATE = 44

# Data creation
features, target = make_regression(
    n_samples = 1000, n_features = 4,
    noise = 0.05, random_state = RANDOM_STATE)

features += 10
target += 400


target = target.reshape(-1, 1)


# Splitting dataset
X_train, X_valid, y_train, y_valid = train_test_split(
  features, target, test_size=0.2, random_state=RANDOM_STATE)

X_valid, X_test, y_valid, y_test = train_test_split(
  X_valid, y_valid, test_size=0.5, random_state=RANDOM_STATE)


x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)


def transformer(inputs: list, scaler) -> list:
  to_return = list()
  for item in inputs:
    scaled = scaler.transform(item)
    scaled = torch.tensor(scaled, dtype=torch.float)
    to_return.append(scaled)
  return to_return


X_train, X_test, X_valid = transformer(
  [X_train, X_test, X_valid], x_scaler)

y_train, y_test, y_valid = transformer(
  [y_train, y_test, y_valid], y_scaler)



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



model = NeuralNet() # creating an instance of the NeuralNet object
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 10_000



for epoch in range(epochs):
  x_split = X_train.split(16)
  y_split = y_train.split(16)

  for x_inst, y_inst in zip(x_split, y_split):
    y_hat = model.forward(x_inst)
    loss = criterion(y_hat, y_inst)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  with torch.no_grad():
    y_hat_val = model(X_valid)
    loss = criterion(y_hat_val, y_valid)
    print(f"Epoch: {epoch} && Loss {loss}")


# estimation for combinations [1, 18, 32, 68, 99] (X_test)
y_hat = model.forward(X_test[[1, 18, 32, 68, 99]])

# removing the scalation done before
y_hat = y_scaler.inverse_transform(
    y_hat.detach().numpy())

# observed rice production
y_real = y_scaler.inverse_transform(
    y_test[[1, 18, 32, 68, 99]])



