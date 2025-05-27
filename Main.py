import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

class NeuralNetwork:
    def __init__(self, network_config):
        self.network_config = network_config
        self.weights = []
        self.biases = []
        self.X = None
        self.y = None

        for i in range(len(network_config) - 1):
            input_size = network_config[i][1]
            output_size = network_config[i + 1][1]
            weight_matrix = np.random.randn(input_size, output_size) * np.sqrt(2. / (input_size + output_size))
            bias_matrix = np.zeros((1, output_size))
            self.weights.append(weight_matrix)
            self.biases.append(bias_matrix)

    def add_training_data(self, samples):
        inputs, outputs = zip(*samples)
        self.X = np.array(inputs)
        self.y = np.array(outputs)

    def forward(self, X):
        self.layer_inputs = []
        self.layer_outputs = [X]
        for i in range(len(self.weights)):
            input_layer = self.layer_outputs[i]
            z = np.dot(input_layer, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = leaky_relu(z)
            self.layer_inputs.append(z)
            self.layer_outputs.append(a)
        return self.layer_outputs[-1]

    def backward(self, y):
        m = self.X.shape[0]
        self.dweights = []
        self.dbiases = []

        output_error = self.layer_outputs[-1] - y
        output_delta = output_error * sigmoid_derivative(self.layer_outputs[-1])

        self.dweights.append(np.dot(self.layer_outputs[-2].T, output_delta) / m)
        self.dbiases.append(np.sum(output_delta, axis=0, keepdims=True) / m)

        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(output_delta, self.weights[i + 1].T)
            delta = error * leaky_relu_derivative(self.layer_inputs[i])
            self.dweights.insert(0, np.dot(self.layer_outputs[i].T, delta) / m)
            self.dbiases.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
            output_delta = delta

    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.dweights[i]
            self.biases[i] -= learning_rate * self.dbiases[i]

    def detect_dead_neurons(self):
        for i, output in enumerate(self.layer_outputs[:-1]):
            dead_neurons = np.all(output == 0, axis=0)
            if np.any(dead_neurons): pass

    def show_layer_averages(self):
        print("\nLayer output statistics:")
        for i, output in enumerate(self.layer_outputs[1:], start=1):
            avg = np.mean(output)
            min_val = np.min(output)
            max_val = np.max(output)
            print(f"Layer {i} average: {avg:.4f}, min: {min_val:.4f}, max: {max_val:.4f}")

    def train(self, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(self.X)
            self.backward(self.y)
            self.update_weights(learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(self.y - self.layer_outputs[-1]))
                print(f"Epoch {epoch}, Loss: {loss}")
                self.detect_dead_neurons()

    def predict(self, X):
        return self.forward(X)
