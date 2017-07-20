import numpy as np
import tensorflow as tf

a = np.array([5, 6, 7])
print(a.shape)
print(a.shape[0])
permutation = np.random.permutation(a.shape[0])
print(permutation)
print(a[permutation])
print(a[[0, 2, 1]])
permutation = np.random.permutation(10)
print(permutation)

b = np.array([[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]
              ])
print(np.reshape(b, 12))
for x in b:
    print(x)
print([np.reshape(x, 6) for x in b])

c = [1, 2, 3]
print(np.mean(c))

d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
print(np.arange(10))
print(d[:, None])
print(np.arange(10) == d[:, None])

tf.train.exponential_decay