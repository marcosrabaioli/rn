from random import random, randrange
from random import seed
from math import exp
from csv import reader

# Initialize a network
# n_inputs: numero de entradas
# n_hidden: numero de neuronios da camada escondida
# n_outputs: numero de neuronios da camada de saida (relacionado com o numero de classes a classificar)
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
# weigths: pesos do neuronio
# inputs: entradas do neuronio
def activate(weights, inputs):
    activation = weights[-1]*1 # Soma o bias
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] # Soma peso*entrada
    return activation

# Transfer neuron activation
# activation: valor de ativação do neuronio
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
# network: rede neural
# row: entrada da rede
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
# output: saida do neuronio
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
# network: rede
# expected: valor de saida esperado
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        # Se não for a ultima camada (camada de saida)
        if i != len(network)-1:
            # Calcula o erro para as demais camadas
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            # Calcula erro pra camada de saida para todos neuronios
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
# network: rede
# row: amostra de treinamento
# l_rate: taxa de aprendizado
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        # Inicializa vetor de entrada com as entradas da rede
        inputs = row[:-1]
        # Se não for a camada de entrada coloca a saida da camada anterior como entrada da proxima
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']*1 # bias

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
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))



# seed(1)
#
#
#
#
#
# # Conjunto de dados para treinar a rede
# dataset = [[2.7810836,2.550537003,0],
#            [1.465489372,2.362125076,0],
#            [3.396561688,4.400293529,0],
#            [1.38807019,1.850220317,0],
#            [3.06407232,3.005305973,0],
#            [7.627531214,2.759262235,1],
#            [5.332441248,2.088626775,1],
#            [6.922596716,1.77106367,1],
#            [8.675418651,-0.242068655,1],
#            [7.673756466,3.508563011,1]]
# # calcula tamanho da entrada
# n_inputs = len(dataset[0]) - 1
# # calcula quantas classes de saida existem
# n_outputs = len(set([row[-1] for row in dataset]))
# # Inicializa a rede
# network = initialize_network(n_inputs, 2, n_outputs)
# # treina a rede
# train_network(network, dataset, 0.5, 20, n_outputs)
# for layer in network:
#     print(layer)
#
# for row in dataset:
#     prediction = predict(network, row)
#     print('Expected=%d, Got=%d' % (row[-1], prediction))

def load_csv(filename):

    dataset = []
    file = open(filename, 'r')
    csv_reader = reader(file)

    for row in csv_reader:
        dataset.append([float(col) for col in row[0].split('\t')])
        dataset[-1][-1] = int(dataset[-1][-1])

    return dataset


def dataset_minmax(dataset):

    minmax = [[min(col), max(col)] for col in zip(*dataset)]
    return minmax

def normalize_dataset(dataset):

    minmax = dataset_minmax(dataset)
    dataset_normalize = []

    for row in dataset:
        row_normalize = [(row[col] - minmax[col][0])/(minmax[col][1] - minmax[col][0]) for col in range(0, len(row)-1)]
        row_normalize.append(row[-1])
        dataset_normalize.append(row_normalize)

    return dataset_normalize

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

dataset = load_csv('clound.csv')
dataset = normalize_dataset(dataset)
dataset_split = cross_validation_split(dataset, 5)

dataset_train = dataset_split[0]



n_inputs = len(dataset_train[0])-1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 10, n_outputs)

train_network(network, dataset_train, 0.1, 1000, n_outputs)

ok = 0
for row in dataset:
    prediction = predict(network, row)
    if row[-1] == prediction:
        ok += 1

print('Mean Accuracy: ', (ok/len(dataset))*100)

