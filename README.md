# Neural Network from Scratch!

This project implements a fully connected neural network in Python, trained on the MNIST digit classification task.  
It includes custom **ReLU** and **Softmax** activations, a **Layer** abstraction, and a **NeuralNetwork** class.  
The code illustrates every essential step: **forward pass**, **backpropagation**, and **gradient update**—all from scratch without high-level deep learning libraries.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Key Components](#key-components)  
   - [Activation Functions](#activation-functions)  
   - [Layer Class](#layer-class)  
   - [NeuralNetwork Class](#neuralnetwork-class)  
3. [Mathematical Underpinnings](#mathematical-underpinnings)  
   - [Forward Pass](#1-forward-pass)  
   - [Loss Function](#2-loss-function)  
   - [Backpropagation](#3-backpropagation)  
   - [Gradient Update](#4-gradient-update)  
4. [Code Structure](#code-structure)  
5. [How to Run](#how-to-run)  
6. [Future Improvements](#future-improvements)

---

## Project Overview

**Goal**: Classify handwritten digits (0–9) from the MNIST dataset.

- **Input dimension**: 784 (each \(28 \times 28\) image flattened).  
- **Hidden layers**: ReLU activation.  
- **Output layer**: Softmax for 10-digit classification.  
- **Training**: Mini-batch gradient descent, cross-entropy loss, multiple epochs.

The code is structured to be understandable and extensible—no PyTorch/TensorFlow dependencies, just raw NumPy.

---

## Key Components

### Activation Functions

1. **ReLU**  
   \[
   \mathrm{ReLU}(z) = \max(0, z)
   \]  
   - Returns 0 for negative inputs, identity for positive inputs.  
   - Derivative is 1 for \(z > 0\), 0 otherwise.

2. **Softmax**  
   \[
   \mathrm{Softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
   \]  
   - Used in the final (output) layer to convert raw scores to probabilities.

### Layer Class

Each layer (`Layer.py`):
- Holds **weights** (W) and **biases** (b) (except for the **input** layer).
- On `forward(...)`, computes:

Z = X @ W + b A = activation(Z)

- Stores an **error vector** (delta) and **gradient matrix** for backprop.

Three types:
1. **INPUT** – Passes data along, no weights/biases.  
2. **HIDDEN** – Uses ReLU activation.  
3. **OUTPUT** – Uses Softmax activation.

### NeuralNetwork Class

- Builds a stack of `Layer` objects: **Input → Hidden(s) → Output**.
- `predict(X)`: Forward pass through all layers.  
- `backProp(y_batch)`: 
1. Compute final error at output layer.  
2. Propagate errors backward to hidden layers.  
3. Compute weight/bias gradients.  

---

## Mathematical Underpinnings

### 1. Forward Pass

For layer \(l\) (not the input):
Z^(l) = A^(l-1) * W^(l) + b^(l) A^(l) = activation( Z^(l) )

- \(A^(l-1)\): activation from previous layer  
- \(W^(l)\), \(b^(l)\): current layer's weights/biases  
- `activation(...)` could be ReLU or Softmax (for output).

### 2. Loss Function

We use **cross-entropy** for classification:
C = - (1/m) * sum_{k=1..m} [ sum_{i=1..num_classes} ( Y_{k,i} * log(Y_hat_{k,i}) ) ]

- \(m\): batch size  
- \(Y_{k}\): one-hot true label for sample k  
- \(Y_hat_{k}\): predicted probability (from Softmax).

### 3. Backpropagation

1. **Output layer error** (\(\delta^{(output)}\)):  
delta_output = A_output - Y

when using softmax + cross-entropy.

2. **Hidden layer error**:  
- `∘` is the elementwise (Hadamard) product.  
- `d(ReLU(Z))` is 1 if Z>0, else 0.

### 4. Gradient Update

The gradient w.r.t. layer \(l\)'s weights:
grad(W^(l)) = (A^(l-1))^T * delta^(l)


To update:
W^(l) <- W^(l) - η * grad(W^(l)) b^(l) <- b^(l) - η * sum_of(delta^(l))

for all samples in the batch (either sum or average the gradients).

---

## Code Structure

- **`Activation.py`**  
  - `Softmax.activate(...)` — stable softmax  
  - `ReLU.activate(...)` — ReLU  
  - `ReLU.getActiveNeurons(...)` — returns a mask (1 or 0)

- **`Layer.py`**  
  - `Layer.__init__`  
  - `Layer.forward(X)` — forward pass  
  - `Layer.updateValues(learning_rate)` — weight/bias updates

- **`Network.py`**  
  - `NeuralNetwork.__init__` — creates input, hidden, and output layers  
  - `NeuralNetwork.predict(X)` — forward pass through all layers  
  - `NeuralNetwork.backProp(y_batch)` — calculates deltas and gradients

- **`main.py`**  
  - Loads MNIST data  
  - Performs training loop: forward, backprop, update  
  - Reports loss and accuracy

- **`Settings.py`**  
  - Hyperparameters (e.g., `LEARNING_RATE`, `BATCH_SIZE`, `EPOCHS`)

---

## How to Run

1. **Ensure MNIST data** files (`train-images-idx3-ubyte.gz`, etc.) are in `datasets/`.  
2. Install dependencies (if needed):
   ```bash
   pip install numpy











# Update Log 8 - 07 January 2025
1. Removed Softmax.py
2. Fixed some bugs with naming conventions
3. Added Main.py
4. Added Datasets and fixed some bugs with naming conventions and parameters
5. Added Epochs and Batch Size to Settings.py and epoch and batch loss to main.py
6. Fixing bugs regarding matrix manipulation of finding the error terms
7. Fixing Logic from Converting Schoastic Gradient Descent to Mini-Batch Gradient Descent
8. Identified Bug - Final Error Term represents all the error terms of each training 
9. Need to change from 1 to 64 (batch size) Row Dimension

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