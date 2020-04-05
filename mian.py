import numpy as np
from NeuralNetwork import NeuralNetwork
from random import random



if __name__ == "__main__":
    # print("How many neurons in hidden layer? (1 - 3)")
    hidden_neurons = 3 # input()
    # print("With or without bias? (0 - 1)")
    is_bias = 1 # input()
    if (is_bias == 0):
        is_bias = False
    else:
        is_bias = True

    network = NeuralNetwork(4, hidden_neurons, 4, is_bias)
    # weights = randVector(int(network.hidden_nodes), is_bias)

    training_input = np.genfromtxt("training_input.txt")

    # TESTING
    print(network.weigts_ih)
    network.feedForward(training_input)