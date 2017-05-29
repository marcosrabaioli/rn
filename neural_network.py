from random import random, randrange
from math import exp
from csv import reader
import csv
import pickle


def initialize_network(n_inputs, design, n_outputs):
    """ Inicializa a rede neural.
    
        Inicializa os pesos sinapticos de cada neuronio de cada camada. A tecnica de inicializacao eh utilizar um valor 
        aleatorio entre 0 e 1.
        A arquiteura da rede eh definida pelas variaveis design e n_output.
        Ex:
        
        >>> n_inputs = 10
        >>> design = [8,5]
        >>> n_outputs = 2
        >>> network = initialize_network(n_inputs,design,n_outputs)
        >>> print(network)
        
        Resultara em uma arquitetura 8:5:2, ou seja, uma rede com uma camada de entrada (possuindo 8 neuronios), uma 
        camada escondida (possuindo 5 neuronios) e e uma camada de saida (possuindo 2 neuronios).
            
        @param n_inputs: numero de entradas da rede neural.
        @type n_inputs: int
        @param design: lista que contem a quantidade de neuronios das camadas de entrada e escondidas.
        @type design: [int,int,...]
        @param n_outputs: numero de saidas da rede neural.
        @type n_outputs: int
        @return: Retorna a rede neural com suas camadas, neuronios e respectivos pesos sinapticos.
        @rtype: [[{"wheigths":[float,...]}],...]
    """

    network = list()
    input_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(design[0])]
    network.append(input_layer)

    for layer in range(1, len(design)):
        hiden_layer = [{'weights': [random() for i in range(design[layer-1] + 1)]} for i in range(design[layer])]
        network.append(hiden_layer)

    output_layer = [{'weights': [random() for i in range(design[-1] + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    """ Calcula a ativacao do neuronio.
    
        Calcula o somatorio das entradas multiplicadas pelos respectivos pesos do neuronio, o que eh chamado de ativacao 
        do neuronio.
    
        @param weights: Pesos do neuronio
        @type weights: [float,...]
        @param inputs: Entradas do neuronio
        @type inputs: [float,...]
        @return: Retorna o valor da ativacao do neuronio
        @rtype: float
    """

    activation = weights[-1]*1 # Soma o bias
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] # Soma peso*entrada
    return activation


def transfer(activation):
    """ Calcula o valor da funcao de transferencia da ativacao do neuronio.
    
        A funcao de transferencia eh definida como:
        f(a) =      1
               ------------
                1 + e^(-a)
                
        onde:
            a = ativacao do neuronio
    
    @param activation: Ativacao do neuronio 
    @type activation: float
    @return: Retorna o valor da funcao de transferencia da ativacao do neuronio
    """
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    """ Propagacao das entradas da rede para a saida da rede.
        
        Executa a propagacao das entradas da rede ate a saida da rede.    
        @param network: Rede neural
        @type network: network
        @param row: Entradas da rede neural
        @type row: [float,...]
        @return: Retorna a saida dos neuronios de saida da rede
        @rtype: [float,...]
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def transfer_derivative(output):
    """ Calcula a derivada da saida do neuronio.
    
        A derivada da funcao logistica eh definida como:
        f(x) =   1
              -------
               1 - x
        onde:
            x = saida do neuronio
    
        @param output: Saida do neuronio
        @type output: float
        @return: Retorna a derivada da saida do neuronio
        @rtype: float
    """
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    """ Backpropagation do erro.
    
        Faz a propagacao do erro da saida para a entrada da rede.
    
        @param network: Rede neural
        @type network: network
        @param expected: Valor esperado na saida da rede
        @type expected: float
    """

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

        # Calcula erro*derivada da saida para todos neuronios
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    """ Atualiza os pesos dos neuronios a partir do erro.
        
        @param network: Rede neural
        @type network: network
        @param row: Amostra de treinamento
        @type row: [float,...]
        @param l_rate: Taxa de aprendizado
        @type l_rate: float
    """

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

def train_network(network, train, l_rate, n_epoch, n_outputs):
    """ Treina a rede neural utilizando um numero fixo de epocas.
    
        @param network: Rede neural
        @type network: network
        @param train: Amostra de treinamento
        @type train: [[float,...],...]
        @param l_rate: Taxa de aprendixado
        @type l_rate: float
        @param n_epoch: Numero de epocas
        @type n_epoch: int
        @param n_outputs: Numero de classes de saida
        @type n_outputs: int
    """

    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    """ Calcula a predicao da rede neural.
    
        @param network: Rede neural
        @type network: network
        @param row: Amostra de entrada
        @type row: [float,...]
        @return: Retorna a predicao da rede, indice de qual neuronio foi mais ativado
        @type: int
    """

    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


def load_csv(filename):
    """ Carrega dataset em csv.
    
        @param filename: Caminho de destino do arquivo csv
        @type filename: string
        @return: Retorna uma lista com os dados do dataset
        @rtype: [[float,...],...]
    """

    dataset = []
    file = open(filename, 'r')
    csv_reader = reader(file)

    for row in csv_reader:
        dataset.append([float(col) for col in row[0].split('\t')])
        dataset[-1][-1] = int(dataset[-1][-1])

    return dataset


def dataset_minmax(dataset):
    """ Identifica maior e menor valor para cada coluna do dataset.
    
        @param dataset: Conjunto de dados
        @type dataset: [[float,...],...]
        @return: Retorna uma lista com os valores maximos e minimos de cada coluna
        @rtype: [[float,float],...]
    """

    minmax = [[min(col), max(col)] for col in zip(*dataset)]
    return minmax

def normalize_dataset(dataset):
    """ Normaliza o conjunto de dados.
    
        @param dataset: Conjunto de dados 
        @type dataset: [[float,...],...]
        @return: Retorna o conjunto de dados normalizado
        @rtype: [[float,...],...]
    """

    minmax = dataset_minmax(dataset)
    dataset_normalize = []

    for row in dataset:
        row_normalize = [(row[col] - minmax[col][0])/(minmax[col][1] - minmax[col][0]) for col in range(0, len(row)-1)]
        row_normalize.append(row[-1])
        dataset_normalize.append(row_normalize)

    return dataset_normalize

def cross_validation_split(dataset, n_folds):
    """ Separa o conjunto de dados em grupos.
    
        A ideia eh separar o conjunto de dados em partes iguais para realizar o treinamento, teste e validacao do sistema.
    
        @param dataset: Conjunto de dados 
        @type dataset: [[float,...],...]
        @param n_folds: Numero de grupos para separacao
        @type n_folds: int
        @return: Retorna uma lista com n_folds posicoes contendo os grupos de conjutos de dados
    """

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

def select_dataset_train(dataset, n_folds):
    """ Separa o conjunto de dados em grupos contendo a mesma quantidade de cada classe de saida.
    
        @param dataset: Conjunto de dados 
        @type dataset: [[float,...],...]
        @param n_folds: Numero de grupos para separacao
        @type n_folds: int
        @return: Retorna uma lista com n_folds posicoes contendo os grupos de conjutos de dados
    """
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    fold = list()
    while len(fold) < fold_size/2:
        index = randrange(len(dataset_copy))
        if dataset_copy[index][-1] == 1:
            fold.append(dataset_copy.pop(index))

    while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
        if dataset_copy[index][-1] == 0:
            fold.append(dataset_copy.pop(index))

    return fold


dataset = load_csv('arquivos_de_treino/20/10vars/CIOS_20.csv')
dataset = normalize_dataset(dataset)

dataset_split = cross_validation_split(dataset, 3)
dataset_train = dataset_split[0]

n_inputs = len(dataset_train[0])-1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, [9], n_outputs)


train_network(network, dataset, 0.05, 2000, n_outputs)

with open('network.pickle', 'wb') as f:
    pickle.dump(network, f)

ok = 0
pp = 0
nd = 0
fp = 0
nn = 0
cios_reais = 0

csvfile = open('resultado_rn.csv', 'w')
spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

for row in dataset:
    prediction = predict(network, row)
    if row[-1] == prediction:
        ok += 1

    if row[-1]==1 and prediction==1:
        pp +=1
    elif row[-1] == 1 and prediction == 0:
        nd+=1
    elif row[-1] == 0 and prediction == 1:
        fp +=1
    elif row[-1] == 0 and prediction == 0:
        nn +=1

    if row[-1]==1:
        cios_reais +=1
    row.append(prediction)
    spamwriter.writerow(row)

print('Media geral: ', (ok/len(dataset))*100)
print('Cios detectados: ', (pp/cios_reais)*100)
print('Cios nao detectados: ', ((nd)/cios_reais)*100)
print('Falsos positivos: ', (fp/(pp+fp))*100)
print('Negativos: ', (nn/(len(dataset)-cios_reais))*100)