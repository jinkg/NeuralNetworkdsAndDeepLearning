from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import tensorflow as tf

layers = tf.contrib.layers
learn = tf.contrib.learn


def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(
        tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size,10) and
    # with a on-value of 1 for each one-hot vector of length 10
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feture to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels
    feature = tf.reshape(feature, [-1, 3072, 100, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.convolution2d(
            feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)

    # Second conv layer will comput 64 features for each 5x5 path
    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.convolution2d(
            h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 1228800])

    h_fc1 = layers.dropout(
        layers.fully_connected(
            h_pool2_flat, 1024, activation_fn=tf.nn.relu),
        keep_prob=0.5,
        is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    # Create a tensor for training op
    train_op = layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='SGD',
        learning_rate=0.001)

    return tf.argmax(logits, 1), loss, train_op


def load_cifar(file):
    import pickle
    import numpy as np
    with open(file, 'rb') as inf:
        cifar = pickle.load(inf)
    data = cifar['data'].reshape((10000, 3, 32, 32))
    data = np.rollaxis(data, 3, 1)
    data = np.rollaxis(data, 3, 1)
    y = np.array(cifar['labels'])

    # Just get 2s versus 9s to start
    # Remove these lines when you want to build a big model
    mask = (y == 2) | (y == 9)
    data = data[mask]
    y = y[mask]

    return data, y


def main(unused_args):
    # Download and load MNIST dataset.
    X_train, y_train = load_cifar("../../Demo/cifar/cifar-10-batches-py/data_batch_1")
    X_test, y_test = load_cifar("../../Demo/cifar/cifar-10-batches-py/test_batch")
    print(len(X_train))
    # Linear classifier.
    feature_columns = learn.infer_real_valued_columns_from_input(
        X_train)
    classifier = learn.LinearClassifier(
        feature_columns=feature_columns, n_classes=10)
    classifier.fit(X_train,
                   y_train.astype(np.int32),
                   batch_size=100,
                   steps=1000)
    score = metrics.accuracy_score(y_test,
                                   list(classifier.predict(X_test)))

    print('Accuracy: {0:f}'.format(score))

    # Convolutional network
    classifier = learn.Estimator(model_fn=conv_model)
    classifier.fit(X_train.astype(np.float32),
                   y_train,
                   batch_size=100,
                   steps=20000)
    score = metrics.accuracy_score(y_test,
                                   list(classifier.predict(X_test)))

    print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
    tf.app.run()
