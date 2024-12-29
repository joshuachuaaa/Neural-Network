import numpy as np
import Error
from __future__ import annotations
from Activation import ReLU, Softmax

class LayerType(Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    HIDDEN = "Hidden"

class Layer : 

    def __init__( self, inputDim:int, outputDim:int, layerType:LayerType):
        """
        Initialization of Layer Class
        Type of Layer determined by activation type
        """
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.layerType = layerType
        

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        if self.layerType == LayerType.INPUT:
            return
        self.input = X
        self.preActivation = np.dot(X,s)

    def activate(self):
        """Activate Neurons"""
        # Input Layer
        if self.layerType == LayerType.INPUT:
            pass

        # Hidden Layer
        elif self.layerType == LayerType.HIDDEN:
            self.activationVal = ReLU.activate()


            

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
        """Calculate Error Term for Output Layer"""

        if self.is_output:
            self.error_term = np.array(actualValues) - np.array(self.neurons)

        else:
            Error.layerAccess()

    def calcErrorTerm(self):
        """Error Term for Hidden Layers"""










        

        



        