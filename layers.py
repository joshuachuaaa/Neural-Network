import numpy as np
import ReLU
class Layer : 

    def __init__(self, prev_layer_neuron,  num_neurons:int, is_input:bool = False, is_output:bool = False):

        """ To check if layer being created is input or output layer"""
        self.is_input = is_input
        self.is_output = is_output

        """Two dimensional vector for the weights connecting previous layer of neurons to this layer"""
        self.weights = np.random.randn(prev_layer_neuron,num_neurons) * 0.1

        """One dimensional vector for biases"""
        self.biases = np.zeros((1,num_neurons))

        """Previous Layer values"""
        self.input = None

        """Neuron values"""
        self.neuron_value = None

        """Current Layer Values"""
        self.activated_value = None

        """Gradient Vector"""
        self.gradient_vector = np.zeros((1,num_neurons))


    """Calculate the neuron values of the current neuron values"""
    def forward_pass(self, X):

        self.input = X
        
        # Neuron value of previous layer * Weights + Bias
        self.neuron_value = np.dot(self.input, self.weights) + self.biases

        # Activated Value
        self.activated_value = ReLU.activate(self.neuron_value)

    def activate(self):
        if self.is_input:
            pass
        elif self.is_output:
            
        else:
            self.activated_value = ReLU.activate(self.neuron_value)



    def backward_pass(self, gradient_vector):





        

        



        