#!/usr/bin/env python3
"""Implement incremental updates for a Gaussian process."""

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

    def predict(self, X_s):
        """Return predictive means and variances at X_s."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)
        covariance = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(covariance)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Append one sample and recompute the covariance matrix."""
        X_new = np.asarray(X_new).reshape(1, 1)
        Y_new = np.asarray(Y_new).reshape(1, 1)
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
