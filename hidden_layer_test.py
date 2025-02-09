import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def forward_propagation(input_data, w1, w2, w3):
    hidden_layer1 = sigmoid(np.dot(input_data, w1))
    hidden_layer2 = sigmoid(np.dot(hidden_layer1, w2))
    output = sigmoid(np.dot(hidden_layer2, w3))
    return hidden_layer1, hidden_layer2, output


def predict(x_val, w1, w2, w3):
    hidden1 = sigmoid(np.dot(x_val, w1))
    hidden2 = sigmoid(np.dot(hidden1, w2))
    predictions = sigmoid(np.dot(hidden2, w3))
    return predictions


def scheduler(epoch, learning_rate, decay_rate=0.0001):
    """
    A simple learning rate scheduler that implements exponential decay

    Args:
        epoch: Current epoch number
        learning_rate: Initial learning rate
        decay_rate: Rate at which learning rate decays

    Returns:
        Decayed learning rate
    """
    return learning_rate * (1.0 / (1.0 + decay_rate * epoch))

# Data preparation
x = np.array([
    [1, 1, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 0, 1]
])

x_val = np.array([
    [1, 1, 0, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 1]
])

y_true = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 0]]).T
y_val_true = np.array([[1, 0, 1, 0, 0]]).transpose()
initial_learning_rate = 0.1

# Network parameters
np.random.seed(42)
w1 = 2 * np.random.random((4, 4)) - 1
w2 = 2 * np.random.random((4, 4)) - 1
w3 = 2 * np.random.random((4, 1)) - 1
epochs = 10000

# Training loop
for i in range(epochs):
    # Forward propagation
    hidden_layer1, hidden_layer2, y_hat = forward_propagation(x, w1, w2, w3)

    # Backpropagation
    output_error = y_true - y_hat
    output_adjustments = output_error * sigmoid_derivative(y_hat)

    hidden2_error = np.dot(output_adjustments, w3.transpose())
    hidden2_adjustments = hidden2_error * sigmoid_derivative(hidden_layer2)

    hidden1_error = np.dot(hidden2_adjustments, w2.transpose())
    hidden1_adjustments = hidden1_error * sigmoid_derivative(hidden_layer1)

    learning_rate = scheduler(i, initial_learning_rate)

    # Update weights
    w3 += initial_learning_rate * np.dot(hidden_layer2.transpose(), output_adjustments)
    w2 += initial_learning_rate * np.dot(hidden_layer1.transpose(), hidden2_adjustments)
    w1 += initial_learning_rate * np.dot(x.transpose(), hidden1_adjustments)

    # Print progress
    if i % 1000 == 0:
        val_predictions = predict(x_val, w1, w2, w3)
        print(f"Epoch {i}, Predictions: \n{val_predictions}")

print(f"y_val_true: \n{y_val_true}")
