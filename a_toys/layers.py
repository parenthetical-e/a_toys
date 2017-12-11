from a_toys.activation import sigmoid
import autograd.numpy as np


def sigmoid_layer(x, W, b):
    return sigmoid(np.dot(x, W) + b)
