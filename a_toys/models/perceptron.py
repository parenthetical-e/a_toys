from sklearn import datasets

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam

from a_toys.loss import l2
from a_toys.layers import sigmoid_layer
from a_toys.datasets import load_mnist


def _init_params(m, n, scale, seed=None):
    prng = npr.RandomState(seed)

    W0 = prng.randn(m, n) * scale
    b0 = prng.randn(n) * scale

    return W0, b0


def init_perceptron(input_size, output_size, scale, seed=None):
    params = [(_init_params(input_size, input_size, scale)),
              (_init_params(input_size, output_size, scale))]

    def perceptron(X, params):
        z = X
        for l, p in enumerate(params):
            W, b = p
            z = sigmoid_layer(z, W, b)

        return z

    return perceptron, params


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # User init
    scale = 0.1
    batch_size = 256
    num_epochs = 5
    step_size = 0.001

    # ----------------------------------------------------------------------
    # Import data, and split into train/test lists
    N, Xs_train, y_train, Xs_test, y_test = load_mnist()
    img_size = Xs_train.shape[1]
    n_digits = 10

    # ----------------------------------------------------------------------
    # Batching
    num_batches = int(np.ceil(len(Xs_train) / batch_size))

    def batch_index(i):
        idx = i % num_batches
        return slice(idx * batch_size, (idx + 1) * batch_size)

    # ----------------------------------------------------------------------
    # !
    perceptron, params = init_perceptron(img_size, n_digits, scale)

    def objective(params, i):
        idx = batch_index(i)

        X = Xs_train[idx]
        y = y_train[idx]

        y_pred = perceptron(X, params)

        return l2(y, y_pred)

    objective_grad = grad(objective)

    # ----------------------------------------------------------------------
    # Progress callback
    def accuracy(params, inputs, targets):
        target_class = np.argmax(targets, axis=1)
        predicted_class = np.argmax(perceptron(inputs, params), axis=1)

        return np.mean(predicted_class == target_class)

    print("     Epoch     |    Train accuracy  |       Test accuracy  ")

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, Xs_train, y_train)
            test_acc = accuracy(params, Xs_test, y_test)
            print("{:15}|{:20}|{:20}".format(iter // num_batches, train_acc,
                                             test_acc))

    # ----------------------------------------------------------------------
    # !
    opt_params = adam(
        objective_grad,
        params,
        step_size=step_size,
        num_iters=num_epochs * num_batches,
        callback=print_perf)
