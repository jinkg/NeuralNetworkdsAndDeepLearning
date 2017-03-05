import mnist_loader
import mnist_network1

training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()

net = mnist_network1.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
