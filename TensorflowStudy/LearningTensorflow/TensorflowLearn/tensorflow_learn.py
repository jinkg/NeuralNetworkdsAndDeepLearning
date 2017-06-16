from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn import svm
import numpy as np
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics

digits = load_digits()

fig = plt.figure(figsize=(3, 3))

plt.imshow(digits['images'][0], cmap="gray", interpolation='none')

# plt.show()

classifier = svm.SVC(gamma=0.001)
classifier.fit(digits.data, digits.target)
predicted = classifier.predict(digits.data)

print(np.mean(digits.target == predicted))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

n_classes = len(set(y_train))
# classifier = learn.LinearClassifier(feature_columns=[tf.contrib.layers.real_valued_column("",
#                                                                                           dimension=X_train.shape[1])],
#                                     n_classes=n_classes)
classifier = learn.DNNClassifier([64, 32],
                                 feature_columns=[tf.contrib.layers.real_valued_column("", dimension=X_train.shape[1])],
                                 n_classes=n_classes)
classifier.fit(X_train, y_train, steps=20)

y_pred = classifier.predict(X_test)
y_result = list(y_pred)

# print(y_result)
print(np.mean(y_result == y_test))
print(metrics.accuracy_score(y_true=y_test, y_pred=y_result))
