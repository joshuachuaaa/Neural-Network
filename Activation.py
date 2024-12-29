from enum import Enum
import Layer
import numpy as np

class Softmax:
    """
    This class is meant to be used for the final neuron values so that the final activation 
    values of the final layer of the output layer all equates to 1, essentially
    turning the output values into probabilities.
    """
    @staticmethod
    def activate(output_layer : Layer):

        #Neuron Array
        neuron_array = output_layer.neuron_value
 
        #This is to make sure that the values aren't too big when doing the exponentitaion
        adjusted_neuron_values = np.exp(neuron_array - np.max(neuron_array, axis=1, keepdims=True))  # Stability improvement
        
        #Turning the array of neuron values into probabilities
        probabilities = adjusted_neuron_values / np.sum(adjusted_neuron_values, axis=1, keepdims=True)
        
        return probabilities

class ReLU:

    @staticmethod
    def activate(neuron_value):
        """Returns 0 if neuron value < 0"""
        return np.maximium(0, neuron_value)
    

    @staticmethod
    def backward(output_gradient, neuron_value):
        """Computes the gradient of the loss with respect to the input for ReLU."""
        # Use the stored linear_neuron_values from the forward pass
        return output_gradient * (neuron_value > 0)
    



