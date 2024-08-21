import numpy as np
import ReLU
import Softmax

WEIGHT_GRADIENT = 0
BIAS_GRADIENT = 1

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



    def backward_pass(self, downstream_gradient_vector, learning_rate:int):

        if self.is_input:
            pass
        
        elif self.is_output:
            self.get_output_gradient_vector(downstream_gradient_vector)
        
        else:
            
            #ReLU derivative
            ReLU_derivative = self.neuron_value > 0

            #Find the error term
            delta = downstream_gradient_vector * ReLU_derivative
            
            #Calculate the derivative of the loss function with respect to the weights
            self.weight_gradient = np.dot(delta, self.input.T)

            #Calculate the derivative of the loss function with respect to the biases
            self.bias_gradient = delta

            #Update values based on the learning rate 
            self.update_values(learning_rate)

            #Return the gradient for the previous layer -> derivative of the neuron value w.r.t previous layer activation function
            return np.dot(delta,self.weights.T)
        

    
    def update_values(self, learning_rate:int):
        """Update the weights and bias values based on the learning rate"""

        #Update the biases
        self.biases -= learning_rate * self.bias_gradient

        #Update the weights
        self.weights -= learning_rate * self.weight_gradient


    def get_output_gradient_vector(self, actual_values):
        """Gradient vector for the output layer."""

        # Gradient of the loss with respect to the logits (final output neuron values)
        self.weight_gradient = self.activated_value - actual_values






        

        



        