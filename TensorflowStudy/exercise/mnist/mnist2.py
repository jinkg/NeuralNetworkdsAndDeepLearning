from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('../../Demo/MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.truncated_normal([784, 64]))
b1 = tf.Variable(tf.zeros([64]))

W2 = tf.Variable(tf.truncated_normal([64, 10]))
b2 = tf.Variable(tf.zeros([10]))

layer1 = tf.matmul(x, W1) + b1
layer1 = tf.nn.relu(layer1)

y_ = tf.add(tf.matmul(layer1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

pred = tf.nn.softmax(y_)
pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(30000):
        batch = mnist.train.next_batch(128)
        _, loss = sess.run([train_step, cost], feed_dict={x: batch[0], y: batch[1]})
        if step % 500 == 0:
            print('loss is %f', loss)

    print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
