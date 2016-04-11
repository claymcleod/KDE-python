from __future__ import division

import numpy as np
from kernels import get_kernel

class KDE(object):

    def __init__(self, kernel='Gaussian'):
        self.kernel = get_kernel(kernel)
        self.kernel_name = kernel.lower()

    def fit(self, X, bandwidth=1., points=1000):
        self.min = np.min(X)
        self.max = np.max(X)
        self.range = self.max - self.min
        self.n_points = len(X)
        self.domain = np.linspace(self.min-bandwidth-1., self.max+bandwidth+1., points)
        self.Y = np.zeros(points)
        self.k_datapoints = []

        for datapoint in X:
            Y_datapoint = self.kernel(self.domain, datapoint, bandwidth)
            self.k_datapoints.append(Y_datapoint)
            self.Y = np.add(self.Y, Y_datapoint)

        self.Y = self.Y / np.max(self.Y)


    def plot(self, individuals=True, individual_scaling=0.2):
        import matplotlib.pyplot as plt

        if individuals:
            for data in self.k_datapoints:
                plt.plot(self.domain, data * individual_scaling, linestyle='--', color='r')

        plt.plot(self.domain, self.Y, color='b')
