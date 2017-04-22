import tensorflow as tf
import numpy as np

x = np.random.randint(1, 100)


def square(inputs):
    return inputs, tf.multiply(inputs, tf.transpose(inputs))


# example = tf.placeholder(tf.int32)

queue = tf.RandomShuffleQueue(100, dtypes=tf.int32, shapes=(), min_after_dequeue=2)

# enqueue_op = queue.enqueue(random.randint(1, 80))
enqueue_op = queue.enqueue(x)

inputs = queue.dequeue_many(5)

square_op = square(inputs=inputs)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

coord = tf.train.Coordinator()

with tf.Session() as sess:
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    try:
        for step in xrange(100):
            if coord.should_stop():
                print('stop')
                break
            print(sess.run(square_op))
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(enqueue_threads)
