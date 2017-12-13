import autograd.numpy as np


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.tanh(x)


def _logsumexp(X, axis, keepdims=False):
    max_X = np.max(X)
    return max_X + np.log(
        np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))


def softmax(x):
    return x - _logsumexp(x, axis=1, keepdims=True)
