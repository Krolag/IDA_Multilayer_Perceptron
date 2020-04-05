import numpy as np
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    hidden_neurons = input("How many neurons in hidden layer? (1 - 3):\t")
    is_bias =  input("With or without bias? (0 - 1):\t")
    if (is_bias == 0):
        is_bias = False
    else:
        is_bias = True

    network = NeuralNetwork(4, int(hidden_neurons), 4, is_bias)
    training_input = np.genfromtxt("training_input.txt")

    # TESTING
    print(network.feedForward(training_input))