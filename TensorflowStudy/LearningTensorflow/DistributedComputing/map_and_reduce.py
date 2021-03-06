import numpy as np
import tensorflow as tf

# x = tf.placeholder(tf.float32, 100)
# mean = tf.reduce_mean(x)
#
# with tf.Session() as sess:
#     result = sess.run(mean, feed_dict={x: np.random.random(100)})
#     print(result)

cluster = tf.train.ClusterSpec({"my_job": ["localhost:2222", "localhost:2223"]})

x = tf.placeholder(tf.float32, 100)

with tf.device("/job:my_job/task:1"):
    first_batch = tf.slice(x, [0], [50])
    mean1 = tf.reduce_mean(first_batch)

with tf.device("/job:my_job/task:0/gpu:0"):
    second_batch = tf.slice(x, [50], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2

with tf.Session("grpc://localhost:2222") as sess:
    result = sess.run(mean, feed_dict={x: np.random.random(100)})
    print(result)
