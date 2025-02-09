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

def logit(y):
    return np.log(y / (1 - y))

def denormalize(y_normalized, y_original):
    # If you used min-max normalization
    y_min = np.min(y_original)
    y_max = np.max(y_original)
    return y_normalized * (y_max - y_min) + y_min


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

np.random.seed(1)

w: np.random.random() = 2 * np.random.random((3, 1)) - 1
print(w)

# logger.section("Neural Network Initialization")
# logger.print_items([
#     (x, "Training inputs", "blue"),
#     (y_true, "Training outputs", "green"),
#     (w, "Synaptic weights", "yellow")
# ], side_by_side=True)
#
# logger.section("Neural network training results")

for i in range(1000):
    input_layer = x
    y_hat = sigmoid(np.dot(input_layer, w))

    error = y_true_normalized - y_hat

    adjustments = error * sigmoid_derivative(y_hat)

    w += Config.learning_rate * np.dot(input_layer.transpose(), adjustments)
    # Denormalize predictions for logging
    y_hat_denormalized = denormalize(y_hat, y_true)
#     logger.print_array(y_hat_denormalized, f"Result epoch: {i} (actual scale)", "red")
#     # Optionally, you can also show normalized values
#     logger.print_array(y_hat, f"Result epoch: {i} (normalized)", "yellow")
#
# logger.print_array(w, "Synaptic weights after training", "blue")


print(f"weights final: {w}")
