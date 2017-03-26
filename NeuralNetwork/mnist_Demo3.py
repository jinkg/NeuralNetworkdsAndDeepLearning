import mnist_network3
from mnist_network3 import Network
from mnist_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from mnist_network3 import ReLU

training_data, validation_data, test_data = mnist_network3.load_data_shared()
mini_batch_size = 50

# net = Network([
#     FullyConnectedLayer(n_in=784, n_out=100),
#     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1,
#         validation_data, test_data)

# net = Network([
#     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
#                   filter_shape=(20, 1, 5, 5),
#                   poolsize=(2, 2)),
#     FullyConnectedLayer(n_in=20 * 12 * 12, n_out=100),
#     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(training_data, 60, mini_batch_size, 0.1,
#         validation_data, test_data)


net = Network([
    ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                  filter_shape=(20, 1, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                  filter_shape=(40, 20, 5, 5),
                  poolsize=(2, 2),
                  activation_fn=ReLU),
    FullyConnectedLayer(
        n_in=40 * 4 * 4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    FullyConnectedLayer(
        n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
    SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
    mini_batch_size)
net.SGD(training_data, 40, mini_batch_size, 0.03,
        validation_data, test_data)
