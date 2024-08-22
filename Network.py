
class NeuralNetwork:
    """Neural Network class -> Serves as a wrapper for all the layers"""

    def __init__(self, input_dimensions:int, output_dimensions:int, hidden_layers:int):

        #Declare number of neurons in the input layer
        self.input_dimensions = input_dimensions

        #Declare number of neurons in the output layer
        self.output_dimensions = output_dimensions

        #Declare number of hidden layers
        self.hidden_layers = hidden_layers

        