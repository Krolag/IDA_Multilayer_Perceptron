import numpy as np
import math
from random import random

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
        if (self.bias_active == True):
            hidden = hidden + self.bias_h
        hidden = np.array(hidden)
        print(hidden)
        

        # LOTS OF MAGIC!

        # Activation Function

        return 0