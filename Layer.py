import numpy as np
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
        self.layerType = layerType
        self.neuronDim = neuronDim

        if self.layerType is not LayerType.INPUT:

            self.weights = np.random.randn(inputDim, neuronDim) * 0.01
            self.biases = np.zeros(1, neuronDim)

            self.errorVector = np.zeros(1, neuronDim)
            self.gradientMatrix = np.zeros(inputDim, neuronDim)

        # For sake of clarity,
        self.input = None
        self.activatedNeurons = np.zeros(1,neuronDim)
        self.boolActiveNeurons = np.zeros(1,neuronDim)
        

    def forward(self, X):
        """
        X shape: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        # If Input Layer, Do Nothing
        if self.layerType == LayerType.INPUT:
            return X
        
        # Calculate Neuron Value if Hidden or Output Layer
        self.input = X
        preActivatedNeurons = (X @ self.weights) + self.biases
        activatedNeurons = self._activate(preActivatedNeurons)

        # ReLU Mask
        self.boolActiveNeurons = self.getActiveNeurons()
    
        return activatedNeurons


    def _activate(self, preActivatedNeurons):
        """Activate Neurons"""
        
        if self.layerType is LayerType.HIDDEN:
            self.activatedNeurons = ReLU.activate(preActivatedNeurons)

        elif self.layerType is LayerType.OUTPUT:
            self.activatedNeurons = Softmax.activate(preActivatedNeurons)

        return self.activatedNeurons
    
    def getActiveNeurons(self):
        """Return 1 for active neurons and 0 for inactive Neurons"""
        return ReLU.getActiveNeurons(self.activatedNeurons)
        


    def updateValues(self, learning_rate:int):
        """Update the weights and bias values based on the learning rate"""

        #Update the biases
        self.biases -= learning_rate * self.bias_gradient

        #Update the weights
        self.weights -= learning_rate * self.weight_gradient













        

        



        