import Layer
class NeuralNetwork:
    """
    Neural Network class -> Serves as a wrapper for all the layers
    Create input layer -> hidden layers -> output layer
    """

    def __init__(self, input_dims:int, output_dims:int, hidden_layers:int, hidden_layer_dims:int):

        #Declare number of neurons in the input layer
        self.input_dims = input_dims

        #Declare number of neurons in the output layer
        self.output_dims = output_dims

        #Declare number of hidden layers
        self.hidden_layers = hidden_layers

        #Declare number of neurons in hidden layers
        self.hidden_layers_dim = hidden_layer_dims 

        #To Store all the layers within a single array
        self.layer_array = []

        #Create input layer
        self.input_layer: Layer = Layer(None, self.input_dims, is_input=True, is_output=False)

        #Check if any hidden layers are requested
        if hidden_layers != 0:

            #Create and connect hidden layers
            for idx in range(1,hidden_layers):
                
                previous_layer = self.layer_array[idx - 1]
                new_hidden_layer = Layer(previous_layer, self.hidden_layers_dim, is_input=False, is_output=False)
                self.layer_array[idx] = new_hidden_layer
            
        #Create output layer
        self.output_layer : Layer = Layer(self.layer_array[-1], self.output_dims, is_input=False, is_output=True)

    def forward(self):
        """To feed data forward and get the predicted result"""
        pass

    def back_propagate(self):
        """Back propagate the error for training and weight/bias adjustment"""
        pass

