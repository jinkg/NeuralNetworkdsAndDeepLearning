"""
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
"""

import numpy as np
import tensorflow as tf
import operator

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../mnist/MNIST-data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)
Xte, Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
sortedDistIndices = tf.nn.top_k(-distance, k=3).indices
minDistance = tf.gather(distance, sortedDistIndices)

# K-means
k_accuracy = 0.
with tf.Session() as sess:
    for i in range(len(Xte)):

        indices = sess.run(sortedDistIndices, feed_dict={xtr: Xtr, xte: Xte[i]})

        classCount = {}
        for index in range(len(indices)):
            # trueClass = np.argmax(Yte[i])
            predictionClass = np.argmax(Ytr[indices[index]])
            classCount[predictionClass] = classCount.get(predictionClass, 0) + 1
            print("Result", index, "Prediction:", predictionClass,
                  "True Class:", np.argmax(Yte[i]))

        pred = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        if pred == np.argmax(Yte[i]):
            k_accuracy += 1. / len(Xte)

    print("Done!")
    print("K_Accuracy:", k_accuracy)

pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i]})
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),
              "True Class:", np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)

    print("Done!")
    print("Accuracy:", accuracy)

print("Compare:", "k_accuracy", k_accuracy, "accuracy", accuracy)
