# Simple Neural Network Package

A basic neural network implementation in Python using NumPy.  
Supports feedforward networks with Leaky ReLU hidden activations and sigmoid output, training via backpropagation.

## Features

- Configurable network architecture  
- Leaky ReLU activation for hidden layers  
- Sigmoid activation for output layer  
- Forward and backward propagation  
- Detect dead neurons  
- Train and predict methods  

## Installation

Simply copy the package folder `simple_nn` into your project directory.

## Usage

```python
from simple_nn import NeuralNetwork
import numpy as np

# Define network architecture: list of (layer_name, size)
network = NeuralNetwork([('input', 3), ('hidden', 5), ('output', 1)])

# Add training data: list of (input, output) pairs
network.add_training_data([
    ([0, 0, 0], [0]),
    ([0, 1, 0], [1]),
    ([1, 0, 0], [1]),
    ([1, 1, 1], [0]),
])

# Train the network
network.train(1000, learning_rate=0.01)

# Make a prediction
result = network.predict(np.array([[1, 0, 1]]))
print("Prediction:", result)
