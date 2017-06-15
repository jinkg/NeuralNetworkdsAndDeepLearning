import tensorflow as tf

# demo1
x1 = tf.constant(2, dtype=tf.float32)
y1 = tf.constant(9, dtype=tf.float32)
point1 = tf.stack([x1, y1])

x2 = tf.constant(-1, dtype=tf.float32)
y2 = tf.constant(3, dtype=tf.float32)
point2 = tf.stack([x2, y2])

X = tf.transpose(tf.stack([point1, point2]))

with tf.Session() as sess:
    print(sess.run(X))

B = tf.ones((1, 2), dtype=tf.float32)

parameters = tf.matmul(B, tf.matrix_inverse(X))

with tf.Session() as sess:
    A = sess.run(parameters)

b = 1 / A[0][1]
a = -b * A[0][0]
print("Equation: y = {a}x + {b}".format(a=a, b=b))

# demo2
points = tf.constant([[2, 1],
                      [0, 5],
                      [-1, 2]], dtype=tf.float64)

A = tf.constant([
    [2, 1, 1],
    [0, 5, 1],
    [-1, 2, 1]], dtype=tf.float64)

B = -tf.constant([[5], [25], [5]], dtype=tf.float64)

X = tf.matrix_solve(A, B)

with tf.Session() as sess:
    result = sess.run(X)
    D, E, F = result.flatten()

    print("Equation: x**2 + y**2 + {D}x + {E}y + {F} = 0".format(**locals()))
