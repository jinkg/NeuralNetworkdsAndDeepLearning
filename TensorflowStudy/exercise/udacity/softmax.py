import numpy as np
from sklearn.linear_model import LogisticRegression

# np.exp(x) / np.sum(np.exp(x), axis=0)
x = np.array([3.0, 1.0, 0.2])
print(np.exp(x))
print(np.sum(x))

y = np.array([[1, 2, 3, 6],
              [2, 4, 5, 6],
              [3, 8, 7, 6]])
print(np.exp(y))
print(np.sum(y))
print(np.sum(y, axis=0))
print(np.sum(np.exp(y), axis=0))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(x))
print(softmax(x * 10))
print(softmax(x / 10))
