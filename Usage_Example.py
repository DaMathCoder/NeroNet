from NeroNet import *

# Example Usage

network = [
    ['input', 1],     #Network input
    ['hidden1', 2],
    ['output', 1]     #Network output
]

samples = [ #Traning Data: ([In_1, In_2], [Out])
    ([0], [1]),
    ([1], [0]),
]

nn = NeuralNetwork(network)

nn.add_training_data(samples)
nn.train(epochs=75000, learning_rate=0.05)

predictions = nn.predict(nn.X)
print("\nFinal predictions:")
print(predictions)

nn.show_layer_averages()
