import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

x: np.array = np.array([
    [1, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0 , 0, 0]
])

y_true: np.array = np.array([[1, 0, 1, 1, 1, 0]]).T

w: np.random.random() = 2 * np.random.random((4, 1)) - 1 # -1 creates numbers between -1 and 1. without it, it would be between 0 and 2

for i in range(1):
    # input layer ( x )  will be another matrix
    input_layer = x

    # takes the input matrix x and the weights matrix w and uses np.dot to apply matrix multiplication
    # predicted value  will be another matrix
    y_hat = sigmoid(np.dot(input_layer, w))


    # calculated the error between predicted and true values
    # error will be another matrix
    error = y_hat - y_true

    # calculated the adjustment that needs to be made based on the error times the sigmoid derivative of the predicted value
    # adjustments will be another matrix
    adjustments = error * sigmoid_derivative(y_hat)

    print(f"{adjustments}\n")
    print(f"{adjustments.shape}\n")
    # print(f"{input_layer}\n")
    # print(f"original inputlayer: \n{input_layer.shape}\n")
    print(f"transposed inputlayer: \n{input_layer.T.shape}\n")
    # print(f"transposed inputlayer: \n{input_layer.T}\n")

    # update the weights ( use += to actually add instead of replace )
    # np.dot() to apply matrix muliplication to the input layer matrix and the adjustments
    w += 0.001 * np.dot(input_layer.transpose(), adjustments)

    # print(f"error: {error}\n")
    # print(f"adjustments: {adjustments}\n")
    # print(f"final prediction: {y_hat}")
