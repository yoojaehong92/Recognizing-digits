
from math import exp
from random import random
import numpy as np
import os
from struct import unpack
from scipy import misc
from urllib import request
import gzip

# MNIST DataSet URLs
url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_label = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_label = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

urls = {
        'train': {
            'image': url_train_image,
            'label': url_train_label
            },
        'test': {
            'image': url_test_image,
            'label': url_test_label
            }
        }


n_inputs = 100
n_outputs = 10

learning_rate = 0.5
n_epoch = 20


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    # 히든 레이어의 노드의 숫자만큼 히든 레이어를 초기화 한다. 이떄 각 노드에는 인풋 노드의 숫자만큼 웨이트와 1개의 bias가 존재한다.
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    # 아웃풋 레이어의 노드의 숫자만큼 아웃풋 레이어를 초기화 한다. 이떄 각 노드에는 히든 노드의 숫자만큼 웨이트와 1개의 bias가 존재한다.
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1] # bias
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('epoch=%d' % (epoch))


def make_dataset(stage):
    print('start making %s dataset' % (stage))

    dataset = []

    for file in ['image', 'label']:
        if not os.path.isfile(stage + '_' + file):
            print('start downloading %s from %s' % (stage + '_' + file, urls[stage][file]))
            request.urlretrieve(urls[stage][file], stage + '_' + file)
        else:
            print('%s already downloaded' % (stage + '_' + file))

    fp_image = gzip.open(stage + '_image', 'rb')

    fp_label = gzip.open(stage + '_label', 'rb')

    img = np.zeros((28, 28))  # 이미지가 저장될 부분

    # drop header info?
    s = fp_image.read(16)    # read first 16byte
    l = fp_label.read(8)     # read first  8byte

    # read mnist
    while True:
        # 784바이트씩 읽음
        s = fp_image.read(784)
        # 1바이트씩 읽음
        l = fp_label.read(1)

        if not s:
            break
        if not l:
            break
        label = int(l[0])
        # unpack
        img = np.reshape(unpack(len(s) * 'B', s), (28, 28))
        # resize from 28x28 to 10x10
        resized = misc.imresize(img, (10, 10))

        datum = list(resized.flat)
        datum.append(label)

        dataset.append(datum)

    print('complete resizing MNIST dataset to 10 x 10 dataset')

    return dataset


# Divide pixel value by 255, so we can get normalized value that has 0 ~ 1
def normalize_dataset(dataset):
    print('start normalizing data')
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = row[i] / 255


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def train():
    n_hidden = 10

    dataset = make_dataset('train')
    normalize_dataset(dataset)

    os.makedirs('trained_networks', exist_ok=True)
    for i in range(30):
        # filename : numberOfHiddenNodes_LearningRate_numberOfEpoch
        filename = '%d_%.1f_%d' % (n_hidden + i, learning_rate, n_epoch)
        if not os.path.isfile('trained_networks/' + filename):
            print('start training : %s', filename)
            network = initialize_network(n_inputs, n_hidden + i, n_outputs)
            train_network(network, dataset, learning_rate, n_epoch, n_outputs)
            f = open('trained_networks/' + filename, 'w')
            for layer in network:
                f.write(str(layer))
                f.write(',')
            f.close()
        else:
            print(filename + ' is trained already')


def test():
    dataset = make_dataset('test')
    normalize_dataset(dataset)
    network_list = os.listdir('trained_networks')
    accuracy = []
    for filename in network_list:
        true = 0
        f = open('trained_networks/' + filename)
        network = eval(f.read()[:-1])

        for datum in dataset:
            prediction = predict(network, datum)

            if datum[-1] == prediction:
                true += 1

        accuracy.append(true / len(dataset))
        f.close()

    print(accuracy)

    max_value = max(accuracy)
    max_index = accuracy.index(max_value)

    print(network_list[max_index][:2])


if __name__ == "__main__":
    # Test training backprop algorithm

    test()




    
