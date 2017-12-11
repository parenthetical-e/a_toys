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


def init_lenet5(scale, seed=None):
    # params = [(_init_params(input_size, input_size, scale)),
    #   (_init_params(input_size, output_size, scale))]

    def lenet5(X, params):

        return z

    return lenet5, params
