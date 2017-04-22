import tensorflow as tf
import threading


def MyLoop(coord, id):
    i = 0
    while not coord.should_stop():
        print(id)
        i += 1
        if i == 10:
            coord.request_stop()


coord = tf.train.Coordinator()

threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in xrange(10)]

coord.join(threads)

for t in threads:
    t.start()
