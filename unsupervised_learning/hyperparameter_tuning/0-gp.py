#!/usr/bin/env python3
"""Implement a noiseless one-dimensional Gaussian process."""

import numpy as np


class GaussianProcess:
    """Represent a noiseless one-dimensional Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize samples, kernel parameters, and covariance."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Return the RBF covariance matrix between X1 and X2."""
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sqdist = sqdist + np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sqdist = sqdist - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(
            -0.5 / self.l ** 2 * sqdist)
