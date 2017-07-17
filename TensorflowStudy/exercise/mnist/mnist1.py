from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../../Demo/MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

m = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
with tf.Session() as sess:
    sess.run(init)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    # print(sess.run(m, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print(sess.run(y_, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
