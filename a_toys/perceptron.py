from sklearn import datasets

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam

from a_toys.activation import sigmoid
from a_toys.loss import l2
from a_toys.data import load_mnist


def init_params(m, n, scale, seed=None):
    prng = npr.RandomState(seed)

    W0 = prng.randn(m, n) * scale
    b0 = prng.randn(n) * scale

    return W0, b0


def layer(x, W, b):
    return sigmoid(np.dot(x, W) + b)


def iter_accuracy(params, x, y):
    target_class = np.argmax(y, axis=1)
    predicted_class = np.argmax(layer(x, *params), axis=1)

    return np.mean(np.isclose(predicted_class, target_class))


if __name__ == "__main__":

    # User init
    scale = 0.1
    batch_size = 256
    num_epochs = 5
    step_size = 0.001

    m, n = 10, 10

    # Ran init
    W0, b0 = init_params(m, n, scale)

    # Import data, and split into train/test lists
    N, Xs_train, y_train, Xs_test, y_test = load_mnist()

    # Batching
    num_batches = int(np.ceil(len(Xs_train) / batch_size))

    def batch_index(i):
        idx = i % num_batches
        return slice(idx * batch_size, (idx + 1) * batch_size)

    # -
    def objective(params, i):

        print(i)
        idx = batch_index(i)

        X = Xs_train[idx]
        y = y_train[idx]

        y_pred = layer(X, *params)

        return l2(y, y_pred)

    objective_grad = grad(objective)

    # -
    res = adam(
        objective_grad, (W0, b0),
        step_size=step_size,
        num_iters=num_epochs * num_batches)
