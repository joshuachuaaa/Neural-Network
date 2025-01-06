# Neural-Network
My attempt at a neural network implemented in python with no external libraries and built from scratch (with the exception of numpy)


# Update Log 8 - 07 January 2025
1. Removed Softmax.py
2. Fixed some bugs with naming conventions
3. Added Main.py
4. Added Datasets and fixed some bugs with naming conventions and parameters
5. Added Epochs and Batch Size to Settings.py and epoch and batch loss to main.py
6. Fixing bugs regarding matrix manipulation of finding the error terms
7. Fixing Logic from Converting Schoastic Gradient Descent to Mini-Batch Gradient Descent

# Update Log 7 - 05 January 2025
1. Removed repeat code in back prop, added forward propagation logic in neuralnetwork class, added test folder
2. Fixed Neural Network instantiation logic error & Added neuronDim to Layer class to connect the layers
3. Removed Util.py
4. Changed Forward in Neural Network class to Predict(X)
5. Changed gradientVector to be gradientMatrix within layer class, also streamlined logic during back and forward propagation

# Update Log 6 - 02 January 2025
1. Added enumerate in the reversed direction for back propagation, fixed error term calculation
2. Removed getActiveNeurons() & Added boolActiveNeurons - Array of 1 and 0s (1 if activated)
3. Simplified and Fixed backProp logic
4. Added Logic in Layer to store booleanActiveNeurons for easier backProp
5. Updated boolActiveNeuron & activatedNeurons to be np matrices & Added Utils.py
6. Weight Gradient Matrix Calculated -to do biases

# Update Log 5 - 01 January 2025 (Happy New Year!)
1. removed ReLU.py - Moved class to Activation.py
2. Moved logic of retrieving active neuron vector to ReLU using np.where()

# Update Log 5 - 31 Decemember 2024
1. Updated Activation Function to use np.maximium
2. Added Auxiliary functions for activation and finding error terms
3. Moved and decoupled Main Logic of forward and backward propagation to Network class to keep Layer class Small & Modular
4. Added Neuron Vector Calculation in ReLU to determine which neuron was active in the layer (for error term calculation)

# Update Log 4 - 30 December 2024 
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