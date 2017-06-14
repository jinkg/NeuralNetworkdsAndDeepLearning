import tensorflow as tf

# demo1
a = tf.constant(3, name='a')

with tf.Session() as sess:
    print(sess.run(a))

# demo2
a = tf.constant(3, name='a')
b = tf.constant(4, name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# demo3
a = tf.constant([1, 2, 3], name='a')
b = tf.constant([4, 5, 6], name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# demo4
a = tf.constant([1, 2, 3], name='a')
b = tf.constant(4, name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# demo5
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[1, 2, 3], [4, 5, 6]], name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# demo6
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant(100, name='b')
add_op = a + b
with tf.Session() as sess:
    print(sess.run(add_op))

# demo7
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([100, 101, 102], name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))

# demo8, not work
# a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
# b = tf.constant([100, 101, ], name='b')
# add_op = a + b

# with tf.Session() as sess:
#     print(sess.run(add_op))

# demo9
a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant([[100], [101]], name='b')
add_op = a + b

with tf.Session() as sess:
    print(sess.run(add_op))
    print(a.shape)
    print(b.shape)
