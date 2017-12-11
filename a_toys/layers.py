import autograd.numpy as np


def linear_layer(x, W, b):
    return np.dot(x, W) + b
