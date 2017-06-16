import tensorflow as tf
import numpy as np

# demo1
print('demo1')
x = tf.Variable(0, name='x')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    for i in range(5):
        x += 1
        print(sess.run(x))

# demo2
print('demo2')
x = tf.Variable(0, name='x')
threshold = tf.constant(5)

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    while sess.run(tf.less(x, threshold)):
        x += 1
        x_value = sess.run(x)
        print (x_value)

# demo3
print('demo3')
# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([1.0, 2.0], name='w')
# Our model of y = a*x + b
y_model = tf.multiply(x, w[0]) + w[1]

# Our error is defined as the square of the differences
error = tf.square(y - y_model)
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal Tensorflow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

errors = []

with tf.Session() as sess:
    sess.run(model)
    for i in range(1000):
        x_train = tf.random_normal((1,), mean=5, stddev=2.0)
        y_train = x_train * 2 + 6
        x_value, y_value = sess.run([x_train, y_train])
        _, error_value = sess.run([train_op, error], feed_dict={x: x_value, y: y_value})
        errors.append(error_value)

    w_value = sess.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))

import matplotlib.pyplot as plt

plt.plot([np.mean(errors[i - 50:i]) for i in range(len(errors))])
plt.show()
plt.savefig("errors.png")
