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

        #Gradient Vector for weights
        self.weight_gradient = np.zeros((1,num_neurons))

        #Gradient Vector for Biases
        self.bias_gradient = np.zeros((1,num_neurons))

        if prev_layer_neuron:
            #Two dimensional vector for the weights connecting previous layer of neurons to this layer"""
            self.weights = np.random.randn(prev_layer_neuron,num_neurons) * 0.1
        
            #One dimensional vector for biases
            self.biases = np.zeros((1,num_neurons))

            #Previous Layer values
            self.input = prev_layer_neuron



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
            # We have to check for the self.neuron_values > 0 because we applied the ReLU function
            # during the forward pass, which essentially makes the activation value to be 0 which means
            # It did not have any impact on the Loss Function - Acts as a filter / mask
            #Also the derivative of ReLU takes on two values [1,0]
            self.gradient_vector = np.dot(gradient_vector, self.weights.T) * (self.neuron_value > 0)


    

    def get_output_gradient_vector(self, actual_values):
        """
        Gradient vector for the output layer.
        """

        # Gradient of the loss with respect to the logits (final output neuron values)
        self.gradient_vector = self.activated_value - actual_values
        return self.gradient_vector






        

        



        