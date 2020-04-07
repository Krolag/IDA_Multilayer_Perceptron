import numpy as np
import time
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    hidden_neurons = input("How many neurons in hidden layer? (1 - 3):\t")
    is_bias = input("With or without bias? (0 - 1):\t")
    if (is_bias == 0):
        is_bias = False
    else:
        is_bias = True

    network = NeuralNetwork(4, int(hidden_neurons), 4, is_bias)
    training_input = np.genfromtxt("training_input.txt")
    training_output = np.genfromtxt("training_input.txt")

    # TESTING
    start_time = time.time()
    for i in range(2500):
        network.train(training_input, training_output)
    delta_time = (str)("--- %s seconds ---" % (time.time() - start_time))
    output = (network.predict(training_input))
    
    # For Overleaf raport in *.tax:
    result = ''
    for i in range(4):
        for j in range(4):
            if j == 3:
                result = result + str(output[i][j])[:12] + "\\" + '\\'
            else:
                result = result + str(output[i][j])[:12] + " & "
        result += '\n'

    print(result)
    plt.title(delta_time)
    plt.grid()
    plt.xlabel("Epoki")
    plt.ylabel("Błąd")
    plt.plot(network.function)
    plt.show()
    
