from __future__ import print_function

import math
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))

def tophat(x, mu, sig):
    return np.select([sig > np.abs(x - mu)], [1])

def get_kernel(name):
    lowername = name.lower()
    if lowername == 'gaussian':
        return gaussian
    elif lowername == 'tophat':
        return tophat
    elif lowername == 'epanechnikov':
        return epanechnikov
    else:
        raise NameError('Unknown kernel: %s' % name)
