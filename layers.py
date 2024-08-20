import numpy as np
class layer : 
    def __init__(self, prev_layer_neuron,  num_neurons):
        """Two dimensional vector for the weights connecting previous layer of neurons to this layer"""
        self.weights = np.random.randn(prev_layer_neuron,num_neurons) * 0.1

        """One dimensional vector for """
        self.biases = np.zeros((1,num_neurons))

        