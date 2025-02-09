import numpy as np
from rich_logger import RichLogger
from config import Config

logger = RichLogger()


def sigmoid(x: np.array(([]))):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))


def denormalize(y_normalized, y_original):
    y_min = np.min(y_original)
    y_max = np.max(y_original)
    return y_normalized * (y_max - y_min) + y_min


# Input data
x: np.array = np.array(([0, 1, 0],
                        [1, 0, 1],
                        [0, 0, 1],
                        [1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1]))

y_true: np.array = np.array([[3, 7, 2, 9, 4, 6, 1, 12]]).transpose()
y_true_normalized = normalize(y_true)

# Neural Network parameters
np.random.seed(1)
hidden_size = 4  # Size of hidden layer

# Initialize weights with better scaling
w1 = np.random.randn(3, hidden_size) * np.sqrt(2.0 / (3 + hidden_size))  # Input -> Hidden
w2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / (hidden_size + 1))  # Hidden -> Output

# Adjusted learning rate
learning_rate = 0.1

y_hat_denormalized = None
# Training loop
for i in range(Config.epochs):
    input_layer = x
    # Forward propagation
    hidden_layer = sigmoid(np.dot(input_layer, w1))
    y_hat = sigmoid(np.dot(hidden_layer, w2))

    # Backpropagation
    output_error = y_true_normalized - y_hat
    output_delta = output_error * sigmoid_derivative(y_hat)

    hidden_error = output_delta.dot(w2.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)

    # Update weights
    w2 += learning_rate * hidden_layer.T.dot(output_delta)
    w1 += learning_rate * x.T.dot(hidden_delta)

    # Logging
    y_hat_denormalized = denormalize(y_hat, y_true)