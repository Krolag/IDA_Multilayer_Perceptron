import numpy as np
import math
from random import random, seed

seed(2137)

def randVector(rows, columns):
    output = []
    tmp = []
    for iterator in range(rows):
        for i in range(columns):
            tmp.append(random() * 2 - 1)
            if (i == columns - 1):
                output.append(tmp)
                tmp = []
    return np.array(output)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
def map(this_arr, function):
    size = np.shape(this_arr)
    for i in range(size[0]):
        for j in range(size[1]):
            value = this_arr[i][j]
            this_arr[i][j] = function(value)

class NeuralNetwork:
    def __init__(self, numI, numH, numO, is_bias):
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.bias_active = is_bias
        self.bias_h = np.full((self.hidden_nodes, 1), 1)
        self.bias_o = np.full((self.output_nodes, 1), 1)
        self.weigts_ih = randVector(int(self.hidden_nodes), int(self.input_nodes))
        self.weigts_ho = randVector(int(self.output_nodes), int(self.hidden_nodes))

    def feedForward(self, input_array):
        # Generating the Hidden Outputs
        inputs = input_array
        hidden = np.dot(self.weigts_ih, inputs)
        hidden = np.array(hidden)
        if (self.bias_active == True):
            hidden = hidden + self.bias_h
        map(hidden, sigmoid)

        # Generating the Output's Output
        output = np.dot(self.weigts_ho, hidden)
        output = np.array(output)
        if (self.bias_active == True):
            output = output + self.bias_o
        map(output, sigmoid)

        # Sendig it back to the caller
        return output

    def train(self, inputs, answer):
        return 0