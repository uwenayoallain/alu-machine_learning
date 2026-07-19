#!/usr/bin/env python3
"""Represent a multivariate normal distribution."""

import numpy as np


class MultiNormal:
    """Represent a multivariate normal distribution estimated from data."""

    def __init__(self, data):
        """Initialize the distribution using data of shape (d, n)."""
        if type(data) is not np.ndarray or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        centered = data - self.mean
        self.cov = np.matmul(centered, centered.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """Calculate the probability density at data point x."""
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        dimensions = self.mean.shape[0]
        if x.shape != (dimensions, 1):
            message = "x must have the shape ({}, 1)".format(dimensions)
            raise ValueError(message)

        difference = x - self.mean
        determinant = np.linalg.det(self.cov)
        inverse = np.linalg.inv(self.cov)
        exponent = -0.5 * np.matmul(np.matmul(difference.T, inverse),
                                    difference)
        denominator = np.sqrt((2 * np.pi) ** dimensions * determinant)
        return float(np.exp(exponent) / denominator)
