import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print(sess.run(linear_model - y, {x: x_train, y: y_train}))
print(sess.run(squared_deltas, {x: x_train, y: y_train}))
print(sess.run(loss, {x: x_train, y: y_train}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])

print(sess.run(linear_model - y, {x: x_train, y: y_train}))
print(sess.run(squared_deltas, {x: x_train, y: y_train}))
print(sess.run(loss, {x: x_train, y: y_train}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})

print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
