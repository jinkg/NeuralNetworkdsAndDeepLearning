import matplotlib

matplotlib.use('TKAgg')

import tensorflow as tf
from functions import update_board
from matplotlib import pyplot as plt

shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32)

X = [1]
with tf.Session() as sess:
    X[0] = sess.run(initial_board)

fig = plt.figure()
plot = plt.imshow(X[0], cmap='Greys', interpolation='nearest')
# plt.show()

board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])

with tf.Session() as sess:
    initial_board_values = sess.run(initial_board)
    X[0] = sess.run(board_update, feed_dict={board: initial_board_values})[0]

    import matplotlib.animation as animation


    def game_of_life(*args):
        X[0] = sess.run(board_update, feed_dict={board: X[0]})[0]
        plot.set_array(X[0])
        return plot,


    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
    plt.show()
