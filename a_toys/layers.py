import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.scipy.signal import convolve


def conv_layer(X, W_shared, b, axes=([2, 3], [2, 3]), dot_axes=([1], [0])):
    """A convolutaion layer.

    Notes
    -----
    The expected X_shated shape, with the default config is,
        (N_data, 1, n_pix_x, n_pix_x) 
        1: color channel: defaults to grey
    The expected b shape is 
        (1, n_filters, kernel_size, kernel_size)
    """

    conv = convolve(X, W_shared, axes=axes, dot_axes=dot_axes, mode='valid')

    return conv + b


def init_conv_layer(n_filters, kernel_size, scale, seed=None):
    """Initialize parameters for a conv_layer"""
    prng = npr.RandomState(seed)

    n_weights = n_filters * kernel_size[0] * kernel_size[1]
    W_shared = prng.randn(n_weights) * scale
    W_shared = W_shared.reshape((1, n_filters, kernel_size[0], kernel_size[1]))

    b = prng.randn(n_filters) * scale
    b = b.reshape((1, n_filters, 1, 1))

    return W_shared, b


def max_pool_layer(X, pool_shape):
    """A max-pooling layer."""

    new_shape = X.shape[:2]

    # Figure out how to ereshape X so 
    # we can do a 2d max efficiently
    for i in range(0, 2):
        pool_width = pool_shape[i]
        img_width = X.shape[i + 2]
        new_shape += (
            np.int(np.floor(np.array(img_width) / np.array(pool_width))),
            np.int(pool_width))

        # new_shape += (img_width // pool_width, pool_width)

    # Shape
    X_max = X.reshape(new_shape)

    # Max x, then y (in the reshaped view)
    # The result will be match X in shape,
    # except for the pooled dimesions which
    # will be reduced by 1 / shape.
    return np.max(np.max(X_max, axis=3), axis=4)


def init_max_pool_layer(max_width, max_height):
    """Intialize parameters for a max_pool_layer"""
    return (np.float(max_width), np.float(max_height))


def linear_layer(X, W, b):
    """A linear layer"""
    return np.dot(X, W) + b


def init_linear_layer(X_shape, scale, seed=None):
    """Intialize parameters for a linear_layer"""
    prng = npr.RandomState(seed)

    m, n = X_shape
    W = prng.randn(m, n) * scale
    b = prng.randn(n) * scale

    return W, b