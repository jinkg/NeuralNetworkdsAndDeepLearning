import tensorflow  as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
add_and_triple = adder_node * 3

sess = tf.Session()

print(sess.run(adder_node, {a: 3., b: 4}))
print(sess.run(adder_node, {a: [1, 3], b: [4.0, 4.5]}))

print(sess.run(add_and_triple, {a: 1, b: 2}))
print(sess.run(add_and_triple, {a: [1, 2, 3], b: [4, 5, 6]}))
