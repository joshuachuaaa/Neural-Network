from Layer import Layer, LayerType
import Settings
import numpy as np
from typing import List

class NeuralNetwork:
    """
    Neural Network class -> Serves as a wrapper for all the layers
    Create input layer -> hidden layers -> output layer
    """

    def __init__(self):

        # Declare number of neurons in the input & output layers
        self.input_dims = Settings.IN_DIMS
        self.output_dims = Settings.OUT_DIM

        # Declare number of hidden layers & neurons
        self.hidden_layers = Settings.HIDDEN_LAYERS
        self.hidden_layers_dim = Settings.HIDDEN_LAYER_DIM

        # To Store all the layers within a single array
        self.layer_array : List[Layer] = []

        # Create input and add input layer to array
        self.input_layer: Layer = Layer(None, self.input_dims, LayerType.INPUT)
        self.layer_array.append(self.input_layer)

        # Create Hidden Layers
        if self.hidden_layers > 0:

            #Create and connect hidden layers
            for idx in range(1,self.hidden_layers):

                # Get previous layer output Dimensions
                previousLayerDim = self.layer_array[idx - 1].neuronDim

                # Create the new hidden array ( hooking up the previous array )
                new_hidden_layer = Layer(previousLayerDim, self.hidden_layers_dim, LayerType.HIDDEN)

                # Add this Layer to the array
                self.layer_array.append(new_hidden_layer)
            
        # Last Hidden Layer Neuron Dimensions
        previousLayerDim = self.layer_array[idx - 1].neuronDim

        # Creating Output Layer
        self.output_layer : Layer = Layer(previousLayerDim, self.output_dims, LayerType.OUTPUT)

        # Add it to Array
        self.layer_array.append(self.output_layer)

    def predict(self, X):
        """To feed data forward and get the predicted result"""
        out = X
        for layer in self.layer_array:
            out = layer.forward(out)
        return out


    def backProp(self):
        """Back propagate the error for training and weight/bias adjustment"""

        for idx, layer in enumerate(reversed(self.layer_array)):

            # Stops when reaches input
            
            if layer.layerType is LayerType.INPUT or idx == 0:
                return

            # Get reference to previous
            previousLayer = self.layer_array[idx -1]
            
            if layer.layerType is LayerType.OUTPUT:
                layer.errorVector = self._calcFinalError
            
            previousLayer.errorVector = self.calcErrorTerm(previousLayer, layer)

            # Get the Gradient Vector
            layer.gradientMatrix = self._calcGradientVector(layer, previousLayer)
            
        return 
        
    def _calcFinalError(self, rightVals):
        """Find the error term in the output"""
        return rightVals - self.layer_array[-1]
    
    def _calcGradientMatrix(self, layer , prevLayer):
        """Returns the gradient matrix"""
        return layer.errorVector @ np.transpose(prevLayer.boolActiveNeurons)
    
    def calcErrorTerm(self, layer : Layer, nextLayer : Layer):
        """Find Erro term for layer"""
        return ( layer.errorVector @ np.transpose(nextLayer.weights))  * (layer.boolActiveNeurons)


