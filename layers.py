import numpy as np
import ReLU
import Softmax

class Layer : 

    def __init__(self, prev_layer_neuron,  num_neurons:int, is_input:bool = False, is_output:bool = False):

        #To check if layer being created is input or output layer
        self.is_input = is_input
        self.is_output = is_output

        #Neuron values
        self.neuron_value = None

        #Current Layer Values
        self.activated_value = None

        #Gradient Vector
        self.gradient_vector = np.zeros((1,num_neurons))

        if prev_layer_neuron:
            #Two dimensional vector for the weights connecting previous layer of neurons to this layer"""
            self.weights = np.random.randn(prev_layer_neuron,num_neurons) * 0.1
        
            #One dimensional vector for biases
            self.biases = np.zeros((1,num_neurons))

            #Previous Layer values
            self.input = None



    def forward_pass(self, X):
        """Calculate the neuron values of the current neuron values"""

        self.input = X
        
        # Neuron value of previous layer * Weights + Bias
        self.neuron_value = np.dot(self.input, self.weights) + self.biases

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
            self.activated_value = ReLU.activate(self.neuron_value)



    def backward_pass(self, gradient_vector):

        if self.is_input:
            pass
        
        elif self.is_output:
            self.get_output_gradient_vector(gradient_vector)
        
        else:


    

    def get_output_gradient_vector(self, actual_values):
        """
        Gradient vector for the output layer.
        """

        # Gradient of the loss with respect to the logits (final output neuron values)
        self.gradient_vector = self.activated_value - actual_values
        return self.gradient_vector






        

        



        