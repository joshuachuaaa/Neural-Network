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
        self.weights = np.random.randn(inputDim, outputDim) * 0.01
        self.biases = np.zeros((1, outputDim,))
        self.layerType = layerType

        # For Clarity Sake,
        self.input = None
        self.preActivation = None
        self.neurons = None
        

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # If Input Layer, Do Nothing
        if self.layerType == LayerType.INPUT:
            return
        
        # Calculate Neuron Value if Hidden or Output Layer
        self.input = X
        preActivation = np.dot(X, self.weights) + self.biases
        return self._activate(preActivation)


    def _activate(self, preActivation):
        """Activate Neurons"""
        
        if self.layerType is LayerType.HIDDEN:
            self.neurons = ReLU.activate(preActivation)

        elif self.layerType is LayerType.OUTPUT:
            self.neurons = Softmax.activate(preActivation)

        return self.neurons


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










        

        



        