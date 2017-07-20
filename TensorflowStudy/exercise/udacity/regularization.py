import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def neural_network(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(hidden_layer_2, weights['h3']), biases['b3'])
    hidden_layer_3 = tf.nn.relu(layer_3)

    output_layer = tf.matmul(hidden_layer_3, weights['out']) + biases['out']
    return output_layer


def neural_network_dropout(x, weights, biases):
    keep_prob = tf.Variable(0.5, tf.float32)

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(layer_1)
    drop_1 = tf.nn.dropout(hidden_layer_1, keep_prob)
    layer_2 = tf.add(tf.matmul(drop_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(layer_2)
    drop_2 = tf.nn.dropout(hidden_layer_2, keep_prob)
    layer_3 = tf.add(tf.matmul(drop_2, weights['h3']), biases['b3'])
    hidden_layer_3 = tf.nn.relu(layer_3)
    drop_3 = tf.nn.dropout(hidden_layer_3, keep_prob)

    out_layer = tf.matmul(drop_3, weights['out']) + biases['out']
    return out_layer


graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, 784))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_hidden_3 = 205
    n_input = 784
    n_classes = 10

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    logits = neural_network_dropout(tf_train_dataset, weights, biases)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)) \
           + 0.01 * tf.nn.l2_loss(weights['h1']) \
           + 0.01 * tf.nn.l2_loss(weights['h2']) \
           + 0.01 * tf.nn.l2_loss(weights['h3']) \
           + 0.01 * tf.nn.l2_loss(weights['out'])

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    logits1 = neural_network(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(logits1)
    logits2 = neural_network(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(logits2)

train_subset = 10000
batch_size = 100
num_steps = 801


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        _, l, predictions = sess.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
            # print(weights['out'].eval())
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
