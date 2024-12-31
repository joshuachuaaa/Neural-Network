# Neural-Network
My attempt at a neural network implemented in python with no external libraries and built from scratch (with the exception of numpy)


# Update Log 5 - 31 Decemember 2024
1. Updated Activation Function to use np.maximium
2. Added Auxiliary functions for activation and finding error terms
3. Moved and decoupled Main Logic of forward and backward propagation to Network class to keep Layer class Small & Modular

# Update Log 4 - 30 December 2024 (Happy New Year!)
1. Added Error Term Matrix to Layer class
2. Added Error.py and functionality to initialize the error term in the output layer
3. Added Pointer for next layer in Layer Class
4. Updated Layer to not store any pointers to previous and next layers
5. Added Enum for Layer Types & Moved Activation Functions to a single File
6. ReFactored Forward Propagation Logic - to Make it cleaner and more modular
7. Consolidated and simplified Forward Propgation Logic

# Update Log 4 - 28 December 20
1. Updated Specificity of Recieving Input Neuron Layer

# Update Log 3 - 12 November 2024
1. Removed Backpropagation Logic

# Update Log 2 - 8 November 2024
1. Fixed Parameter passed in softmax function: output_layer

# Update Log 1 - 2 November 2024
1. Fixed out of bounds error in the creation of the array
2. Changed the logic of the instantiation of the Network class
3. Introduced a Settings.py which serves as a file that contains static variables to decouple the logic and tuning of the neural network.
4. Added Input and Output Layer to layer array.