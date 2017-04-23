import tensorflow as tf


def my_image_filter(input_images):
    conv1_weight = tf.Variable(tf.random_normal([5, 5, 32, 32]),
                               name="conv1_weight")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv1 = tf.nn.conv2d(input_images, conv1_weight,
                         strides=[1, 1, 1, 1], padding='SAME')
    relu1 = tf.nn.relu(conv1 + conv1_biases)

    conv2_weight = tf.Variable(tf.random_normal([5, 5, 32, 32]),
                               name="conv2_weight")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    conv2 = tf.nn.conv2d(relu1, conv2_weight,
                         strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2 + conv2_biases)


def conv_relu(input, kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def my_image_filter2(input_images):
    with tf.variable_scope("conv1"):
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        return conv_relu(relu1, [5, 5, 32, 32], [32])


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(10.))

print("v.name = {0}".format(v.name))
assert v.name == "foo/v:0"

with tf.variable_scope("foo1"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo1", reuse=True):
    v1 = tf.get_variable("v", [1])

assert v is v1

with tf.variable_scope("foo2"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])

print("v.name = {0}".format(v.name))
assert v.name == "foo2/bar/v:0"

with tf.variable_scope("foo3"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])

assert v is v1

with tf.variable_scope("root"):
    # At start, the scope is not reusing.
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # Opened a sub-scope, still not reusing.
        assert tf.get_variable_scope().reuse == False
    # Explicitly opened a reusing scope.
    with tf.variable_scope("foo", reuse=True):
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # Now sub-scope inherits the reuse flag.
            assert tf.get_variable_scope().reuse == True
    with tf.variable_scope("foo"):
        assert tf.get_variable_scope().reuse == False
    # Exited the reusing scope, back to a non-reusing one.
    assert tf.get_variable_scope().reuse == False

with tf.variable_scope("foo4") as foo_scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable("w", [1])
with tf.variable_scope(foo_scope, reuse=True):
    v1 = tf.get_variable("v", [1])
    w1 = tf.get_variable("w", [1])

assert v is v1
assert w is w1

with tf.variable_scope("foo5") as foo_scope:
    assert foo_scope.name == "foo5"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo5"

with tf.Session() as sess:
    with tf.variable_scope("foo6", initializer=tf.constant_initializer(0.4)):
        v = tf.get_variable("v", [1])
        sess.run(tf.global_variables_initializer())
        assert v.eval() == 0.4
        w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3))
        sess.run(tf.global_variables_initializer())
        assert w.eval() == .3
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            sess.run(tf.global_variables_initializer())
            assert v.eval() == .4
        with tf.variable_scope("baz", initializer=tf.constant_initializer(.2)):
            v = tf.get_variable("w", [1])
            sess.run(tf.global_variables_initializer())
            assert v.eval() == .2

with tf.variable_scope("foo7"):
    x = 1.0 + tf.get_variable("v", [1])
print("x.op.name = {0}".format(x.op.name))
assert x.op.name == "foo7/add"

with tf.variable_scope("foo8"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        x = 1.0 + v

print("v.name = {0}".format(v.name))
print("x.op.name = {0}".format(x.op.name))
assert v.name == "foo8/v:0"
assert x.op.name == "foo8/bar/add"
