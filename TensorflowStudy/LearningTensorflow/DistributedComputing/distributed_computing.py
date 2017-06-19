import tensorflow as tf

# x = tf.constant(2)
# y1 = x + 300
# y2 = x - 66
# y = y1 + y2
#
# with tf.Session() as sess:
#     result = sess.run(y)
#     print(result)

cluster = tf.train.ClusterSpec({"my_job": ["localhost:2222", "localhost:2223"]})

x = tf.constant(2)

with tf.device("/job:my_job/task:1"):
    y2 = x - 66
with tf.device("/job:my_job/task:0"):
    y1 = x + 300
    y = y1 + y2

with tf.Session("grpc://localhost:2223") as sess:
    result = sess.run(y)
    print(result)
