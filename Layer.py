import numpy as np
import Error
from __future__ import annotations
from Activation import ReLU, Softmax
from enum import Enum

class LayerType(Enum):
    INPUT = "Input"
    OUTPUT = "Output"
    HIDDEN = "Hidden"

class Layer : 

    def __init__( self, inputDim:int, neuronDim:int, layerType:LayerType):
        """
        Initialization of Layer Class
        Type of Layer determined by activation type
        """
        self.weights = np.random.randn(inputDim, neuronDim) * 0.01
        self.biases = np.zeros(1, neuronDim)
        self.layerType = layerType

        self.errorVector = np.zeros(1, neuronDim)
        self.gradientVector = np.zeros(inputDim, neuronDim)

        # For sake of clarity,
        self.input = None
        self.activatedNeurons = None
        

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
        preActivatedNeurons = np.dot(X, self.weights) + self.biases
    
        return self._activate(preActivatedNeurons)


    def _activate(self, preActivatedNeurons):
        """Activate Neurons"""
        
        if self.layerType is LayerType.HIDDEN:
            self.activatedNeurons = ReLU.activate(preActivatedNeurons)

        elif self.layerType is LayerType.OUTPUT:
            self.activatedNeurons = Softmax.activate(preActivatedNeurons)

        return self.neurons


    def updateValues(self, learning_rate:int):
        """Update the weights and bias values based on the learning rate"""

        #Update the biases
        self.biases -= learning_rate * self.bias_gradient

        #Update the weights
        self.weights -= learning_rate * self.weight_gradient













        

        



        