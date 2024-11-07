import Layer
import Settings
class NeuralNetwork:
    """
    Neural Network class -> Serves as a wrapper for all the layers
    Create input layer -> hidden layers -> output layer
    """

    def __init__(self):

        # Declare number of neurons in the input layer
        self.input_dims = Settings.IN_DIMS

        # Declare number of neurons in the output layer
        self.output_dims = Settings.OUT_DIM

        # Declare number of hidden layers
        self.hidden_layers = Settings.HIDDEN_LAYERS

        # Declare number of neurons in hidden layers
        self.hidden_layers_dim = Settings.HIDDEN_LAYER_DIM

        # To Store all the layers within a single array
        self.layer_array = []

        # Create input and add input layer to array
        self.input_layer: Layer = Layer(None, self.input_dims, is_input=True, is_output=False)
        self.layer_array.append(self.input_layer)

        # Create Hidden Layers
        if self.hidden_layers > 0:

            #Create and connect hidden layers
            for idx in range(1,self.hidden_layers):

                # Gain reference to layer in array
                previous_layer = self.layer_array[idx - 1]

                # Create the new hidden array ( hooking up the previous array )
                new_hidden_layer = Layer(previous_layer, self.hidden_layers_dim, is_input=False, is_output=False)
                
                # Add this Layer to the array
                self.layer_array.append(new_hidden_layer)
            
        #Create and add output layer
        self.output_layer : Layer = Layer(self.layer_array[-1], self.output_dims, is_input=False, is_output=True)
        self.layer_array.append(self.output_layer)

    def forward(self):
        """To feed data forward and get the predicted result"""

        pass

    def back_propagate(self):
        """Back propagate the error for training and weight/bias adjustment"""
        pass

