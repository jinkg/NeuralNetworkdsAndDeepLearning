# coding=utf-8
import tensorflow as tf

# No.1.Create a 3-dimensional matrix. What happens if you add a scalar, array or matrix to it?
# No.2.Use tf.shape (it’s an operation) to get a constant’s shape during operation of the graph.
# [1,2,3]
a = tf.constant([[[1, 2, 3], [4, 5, 6]]], name='a')

# [2,1,3]
a1 = tf.constant([[[1, 2, 3]], [[4, 5, 6]]], name='a1')

# [2,3,1]
a2 = tf.constant([[[1], [2], [3]], [[4], [5], [6]]], name='a2')

# [3,2,1]
a3 = tf.constant([[[1], [2]], [[3], [4]], [[5], [6]]], name='a3')

# [3,1,2]
a4 = tf.constant([[[1, 2]], [[3, 4]], [[5, 6]]], name='a4')

# [1,3,2]
a5 = tf.constant([[[1, 2], [3, 4], [5, 6]]], name='a5')

# scalar
b = tf.constant(100, name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))
    print(sess.run(tf.shape(a)))
    print(sess.run(tf.shape(a1)))
    print(sess.run(tf.shape(a2)))
    print(sess.run(tf.shape(a3)))
    print(sess.run(tf.shape(a4)))
    print(sess.run(tf.shape(a5)))

# array [1]
# [1]
b = tf.constant([100], name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# matrix [1,1],[1,2,1],[1,2,3] can add
# [1,1]
b1 = tf.constant([[100]], name='b1')
add_op = a + b1
with tf.Session() as sess:
    print(sess.run(add_op))

# [1,2,1]
b2 = tf.constant([[[100], [101]]], name='b2')
add_op = a + b2
with tf.Session() as sess:
    print(sess.run(add_op))

# [1,2,3]
b3 = tf.constant([[[100, 101, 102], [103, 104, 105]]], name='b3')
add_op = a + b3
with tf.Session() as sess:
    print(sess.run(add_op))
