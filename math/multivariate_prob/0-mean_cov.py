#!/usr/bin/env python3
"""Calculate the mean and covariance of a data set."""

import numpy as np


def mean_cov(X):
    """Return the mean and sample covariance matrix of a data set."""
    if type(X) is not np.ndarray or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    centered = X - mean
    cov = np.matmul(centered.T, centered) / (X.shape[0] - 1)
    return mean, cov
