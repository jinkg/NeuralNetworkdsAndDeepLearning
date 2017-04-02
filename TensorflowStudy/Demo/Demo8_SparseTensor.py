import tensorflow as tf

sparse_tensor = tf.SparseTensor(indices=[[1, 3, 2], [2, 8, 5]],
                                values=[10, 8],
                                dense_shape=[10, 10, 10])

sess = tf.Session()

print(sess.run(sparse_tensor))
