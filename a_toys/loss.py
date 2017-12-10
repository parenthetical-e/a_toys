import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam


def l2(y, y_pred):
    return np.sum(np.power(y - y_pred, 2))
