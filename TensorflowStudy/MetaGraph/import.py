import tensorflow as tf

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph("../Demo/SaveDir/my-model-10000.meta")
    new_saver.restore(sess, "../Demo/SaveDir/my-model-10000")

    labels = tf.constant(0, tf.int32, shape=[100], name="labels")
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_label = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
    logits = tf.get_collection("logits")[0]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label,
                                                            logits=logits, name="xentropy")
    loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train_op = optimizer.minimize(loss)
    sess.run(train_op)
