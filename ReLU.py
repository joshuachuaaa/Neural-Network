from numpy import np
class ReLU:

    """Returns 0 if neuron value < 0"""
    @staticmethod
    def activate(neuron_value):
        
        return np.maximium(0, neuron_value)
    

    """Computes the gradient of the loss with respect to the input for ReLU."""
    @staticmethod
    def backward(output_gradient, neuron_value):

        # Use the stored linear_neuron_values from the forward pass
        return output_gradient * (neuron_value > 0)
    



