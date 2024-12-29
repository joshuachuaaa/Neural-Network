import numpy as np

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
    
