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

def dsigmoid(y):
    # return sigmoid(x) * (1 - sigmoid(x))
    return y * (1 - y)
    
def map(this_arr, function):
    size = np.shape(this_arr)
    result = this_arr
    for i in range(size[0]):
        for j in range(size[1]):
            value = this_arr[i][j]
            result[i][j] = function(value)
    return result

class NeuralNetwork:
    def __init__(self, numI, numH, numO, is_bias):
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.learing_rate = 0.1
        self.bias_active = is_bias
        self.bias_h = np.full((self.hidden_nodes, 1), 1)
        self.bias_o = np.full((self.output_nodes, 1), 1)
        self.weigts_ih = randVector(int(self.hidden_nodes), int(self.input_nodes))
        self.weigts_ho = randVector(int(self.output_nodes), int(self.hidden_nodes))
        self.function = np.array([])

    def predict(self, input_array):
        # Generating the Hidden Outputs
        inputs = input_array
        hidden = np.dot(self.weigts_ih, inputs)
        hidden += self.bias_h * 1.0

        # Activation fuction
        hidden = map(hidden, sigmoid)

        # Generating the output's output
        output = np.dot(self.weigts_ho, hidden)
        output += self.bias_o * 1.0
        output = map(output, sigmoid)

        # Sending back to the caller
        return output

    def train(self, input_array, targets_array):
        # Generating the Hidden Outputs
        inputs = input_array
        hidden = np.dot(self.weigts_ih, inputs)
        if (self.bias_active == True):
            hidden += self.bias_h * 1.0
        hidden = map(hidden, sigmoid)

        # Generating the Output's Output
        outputs = np.dot(self.weigts_ho, hidden)
        if (self.bias_active == True):
            outputs += self.bias_o * 1.0
        outputs = map(outputs, sigmoid)

        # Copy targets_array to targets
        targets = targets_array

        # Calculate the output's error
        # ERROR = TARGETS - OUTPUTS
        output_errors = targets - outputs
        output_error_power = output_errors * output_errors
        self.function = np.append(self.function, np.average(output_error_power))

        # Calculate the gradient
        gradients = map(outputs, dsigmoid)
        gradients = np.multiply(gradients, output_errors)
        gradients = np.multiply(gradients, self.learing_rate)

        # Calculate deltas
        hidden_T = np.transpose(hidden)
        weight_ho_deltas = np.dot(gradients, hidden_T)

        # Adjusting the weights by deltas
        self.weigts_ho += weight_ho_deltas
        # Adjust the bias by its deltas (which is just the gradients)
        self.bias_o = self.bias_o * 1.0 + gradients

        # Calculate the hidden layer errors
        who_T = np.transpose(self.weigts_ho)
        hidden_errors = np.dot(who_T, output_errors)

        # Calculate hidden gradient
        hidden_gradient = map(hidden, dsigmoid)
        hidden_gradient = np.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = np.multiply(hidden_gradient, self.learing_rate)

        # Calculate input->hidden deltas
        inputs_T = np.transpose(inputs)
        weight_ih_deltas = np.dot(hidden_gradient, inputs_T)

        # Adjusting the weights by deltas
        self.weigts_ih += weight_ih_deltas
        # Adjust the bias by its deltas (which is just the gradients)
        self.bias_h = self.bias_h * 1.0 + hidden_gradient

        # Sending it back to the caller
        # return output_errors + 1