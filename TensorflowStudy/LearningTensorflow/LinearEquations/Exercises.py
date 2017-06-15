import tensorflow as tf
from math import sqrt as sqrt

# No.1. Solve for the circle that contains the following three points: P(2,1), Q(0,5), R(-1,2)
A = tf.constant([
    [2, 1, 1],
    [0, 5, 1],
    [-1, 2, 1]
], dtype=tf.float64)

B = -tf.constant([
    [5],
    [25],
    [5]
], dtype=tf.float64)

X = tf.matrix_solve(A, B)

with tf.Session() as sess:
    result = sess.run(X)
    D, E, F = result.flatten()

    print("Equation: x**2 + y**2 + {D}x + {E}y + {F}".format(**locals()))

# No.2. The general form of an ellipse is given below.
# Solve for the following points (five points are needed to solve this equation):
A = tf.constant([
    [0, 0, 8, 0, 1],
    [24, -8 * sqrt(6), 4, -2 * sqrt(6), 1],
    [4, -4 * sqrt(14), -2 * sqrt(14), 2, 1],
    [9, -3 * sqrt(46), -sqrt(46), 3, 1],
    [25, 5 * sqrt(14), sqrt(14), 5, 1]
], dtype=tf.float64)

B = -tf.constant([
    [64],
    [16],
    [56],
    [46],
    [14]
], dtype=tf.float64)

X = tf.matrix_solve(A, B)

with tf.Session() as sess:
    result = sess.run(X)
    B, C, D, E, F = result.flatten()

    print("Equation: x**2 + {B}y**2 + {C}xy + {D}x + {E}y + {F}".format(**locals()))
