import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
from autograd.scipy.signal import convolve

from a_toys.loss import l2

from a_toys.layers import init_linear_layer
from a_toys.layers import init_conv_layer
from a_toys.layers import init_max_pool_layer
from a_toys.layers import linear_layer
from a_toys.layers import conv_layer
from a_toys.layers import max_pool_layer

from a_toys.activation import sigmoid
from a_toys.activation import softmax
from a_toys.activation import relu

from a_toys.datasets import load_mnist


def init_lenet5(scale,
                kernel_size=5,
                n_filter_l0=6,
                n_filter_l2=16,
                X_pool_shape=(2, 2),
                X_shape_l4=(256, 120),
                X_shape_l5=(120, 83),
                X_shape_l6=(83, 10),
                seed=None):
    """Init parameters for lenet5"""

    # Init with a count... just as placeholder
    # (to ease readbility between init_lenet5(.) and lenet5(.))
    params = range(7)

    # Feature map/pooling
    params[0] = init_conv_layer(n_filter_l0, (kernel_size, kernel_size), scale)
    params[1] = init_max_pool_layer(*X_pool_shape)
    params[2] = init_conv_layer(n_filter_l2, (kernel_size, kernel_size), scale)
    params[3] = init_max_pool_layer(*X_pool_shape)

    # Dense layers
    params[4] = init_linear_layer(X_shape_l4, scale)
    params[5] = init_linear_layer(X_shape_l5, scale)

    # Output/classes
    params[6] = init_linear_layer(X_shape_l6, scale)

    return params


def lenet5(X, params):
    # Feature map/pooling
    z = relu(conv_layer(X, *params[0]))
    z = max_pool_layer(z, params[1])
    z = relu(conv_layer(z, *params[2]))
    z = max_pool_layer(z, params[3])
    z = z.reshape((-1, 16 * 4 * 4))  # flatten for dense

    # Dense layers
    z = relu(linear_layer(z, *params[4]))
    z = relu(linear_layer(z, *params[5]))

    # Output/classes
    z = linear_layer(z, *params[6])

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

    params = init_lenet5(scale)

    # ----------------------------------------------------------------------
    # Batching
    num_batches = int(np.ceil(len(Xs_train) / batch_size))

    def batch_index(i):
        idx = i % num_batches
        return list(range(idx * batch_size, (idx + 1) * batch_size))

    def one_hot(x, K):
        """Convert labels from a sequence of int (0..K-1) to a 
        binary matrix (size_x, K)
        """

        return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)

    # ----------------------------------------------------------------------

    def objective(params, i):
        idx = batch_index(i)

        X = Xs_train[idx].reshape((256, 1, 28, 28))
        y = one_hot(y_train[idx], 10)

        y_pred = lenet5(X, params)

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
