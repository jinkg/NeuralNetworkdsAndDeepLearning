from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../../Demo/MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.zeros([784, 64]))
b1 = tf.Variable(tf.zeros([64]))

W2 = tf.Variable(tf.zeros([64, 10]))
b2 = tf.Variable(tf.zeros([10]))

layer1 = tf.matmul(x, W1) + b1
y_ = tf.matmul(layer1, W2) + b2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

    print(sess.run(W1))
    print(sess.run(b1))
    print(sess.run(W2))
    print(sess.run(b2))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print(sess.run(y_, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
