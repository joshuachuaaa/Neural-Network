import numpy as np
import ReLU
import Error
import Softmax
from __future__ import annotations

class Layer : 

    def __init__(self, prevLayer:Layer , nextLayer:Layer,  num_neurons:int, is_input:bool = False, is_output:bool = False):

        #To check if layer being created is input or output layer
        self.is_input = is_input
        self.is_output = is_output

        #Default Neuron Matrix
        self.neurons = None

        #Default Weight Matrix
        self.weights = None

        #Default Bias Matrix
        self.biases = None

        #Default Error Term
        self.error_term = None

        #Gradient Vector for weights
        self.weight_gradient = np.zeros((1,num_neurons))

        #Gradient Vector for Biases
        self.bias_gradient = np.zeros((1,num_neurons))


        if prev_layer_neuron:
            #Two dimensional vector for the weights connecting previous layer of neurons to this layer"""
            self.weights = np.random.randn(prev_layer_neuron,num_neurons) * 0.1
        
            #One dimensional vector for biases
            self.biases = np.zeros((1,num_neurons))

            #Previous Layer values
            self.input = prev_layer_neuron

            
    def forward_pass(self, X):
        """Calculate the neuron values of the current neuron values"""

        self.input = X
        
        # Neuron value of previous layer * Weights + Bias
        self.neurons = np.dot(self.input, self.weights) + self.biases

        # Activate Value
        self.activate()

    def activate(self):
        """Activate Neurons"""

        #If input layer, No activation Function
        if self.is_input:
            pass

        #If output layer, Use Softmax
        elif self.is_output:
            self.activated_value = Softmax.activate(self.neuron_value)

        #If hidden Layer, use ReLU
        else:
            self.neuron_value = ReLU.activate(self.neuron_value)
        

    def updateValues(self, learning_rate:int):
        """Update the weights and bias values based on the learning rate"""

        #Update the biases
        self.biases -= learning_rate * self.bias_gradient

        #Update the weights
        self.weights -= learning_rate * self.weight_gradient

    def initErrorTerm(self, actualValues):
        """Calculate Error Term for Layer"""

        if self.is_output:
            self.error_term = np.array(actualValues) - np.array(self.neurons)

        else:
            Error.layerAccess()

    def calcErrorTerm(self):







        

        



        