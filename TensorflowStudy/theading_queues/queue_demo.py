import tensorflow as tf

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0., 0., 0.],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue(y)

sess = tf.Session()

with sess.as_default():
    init.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()

print(sess.run(q.size()))
print(sess.run(q.dequeue()))
print(sess.run(q.dequeue()))
print(sess.run(q.dequeue()))
