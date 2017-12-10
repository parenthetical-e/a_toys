import autograd.numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.tanh(x)
