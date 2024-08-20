from numpy import np
class ReLU:

    """Returns 0 if neuron value < 0"""
    def forward_ReLU(self, neruron_value):
        return np.maximium(0, neruron_value)





