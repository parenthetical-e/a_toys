from sklearn import datasets

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam

from a_toys.loss import l2
from a_toys.layers import linear_layer
from a_toys.layers import init_linear_layer
from a_toys.activation import sigmoid
from a_toys.datasets import load_mnist


def init_perceptron(input_size, output_size, scale, seed=None):
    params = [(init_linear_layer((input_size, input_size), scale)),
              (init_linear_layer((input_size, output_size), scale))]

    return params


def perceptron(X, params):
    z = sigmoid(linear_layer(X, *params[0]))  # In
    z = sigmoid(linear_layer(z, *params[1]))  # Output

    return z


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
    params = init_perceptron(img_size, n_digits, scale)

    # ----------------------------------------------------------------------
    # Batching
    num_batches = int(np.ceil(len(Xs_train) / batch_size))

    def batch_index(i):
        idx = i % num_batches
        return slice(idx * batch_size, (idx + 1) * batch_size)

    # ----------------------------------------------------------------------

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
