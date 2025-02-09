import numpy as np
from rich_logger import RichLogger
from config import Config


class SimpleNeuralNetwork:
    def __init__(self, input_size: int, logger: RichLogger, learning_rate: Config.learning_rate, epochs: Config.epochs):
        """
        Initialize the neural network with a hidden layer
        """
        self.logger = logger
        # np.random.seed(1)

        # Add hidden layer (e.g., 4 neurons)
        self.hidden_weights = 2 * np.random.random((input_size, 4)) - 1
        self.output_weights = 2 * np.random.random((4, 1)) - 1

        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Activation function"""
        return 1 / (1 + np.exp(-x))

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        return x * (1 - x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.logger.section("Neural Network Initialization")
        self.logger.print_items([
            (X, "Training inputs", "blue"),
            (y, "Training outputs", "green"),
            (self.hidden_weights, "Initial hidden weights", "yellow"),
            (self.output_weights, "Initial output weights", "yellow")
        ], side_by_side=True)

        self.logger.section("Neural network training results")

        for i in range(self.epochs):
            # Forward propagation
            hidden_layer = self.sigmoid(np.dot(X, self.hidden_weights))
            output_layer = self.sigmoid(np.dot(hidden_layer, self.output_weights))

            # Backpropagation
            output_error = y - output_layer
            output_delta = output_error * self.sigmoid_derivative(output_layer)

            hidden_error = np.dot(output_delta, self.output_weights.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer)

            # Update weights
            self.output_weights += self.learning_rate * np.dot(hidden_layer.T, output_delta)
            self.hidden_weights += self.learning_rate * np.dot(X.T, hidden_delta)

            if i % 10 == 0:  # Print every 10 epochs
                self.logger.print_array(output_layer, f"Result epoch: {i}", "red")

        self.logger.print_array(self.output_weights, "Output weights after training", "blue")
        self.logger.print_array(self.hidden_weights, "Hidden weights after training", "blue")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network
        """
        hidden_layer = self.sigmoid(np.dot(X, self.hidden_weights))
        return self.sigmoid(np.dot(hidden_layer, self.output_weights))