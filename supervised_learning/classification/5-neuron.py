#!/usr/bin/env python3
"""Defines a trainable neuron."""

import numpy as np


class Neuron:
    """Represents a sigmoid neuron trained by gradient descent."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return the weights."""
        return self.__W

    @property
    def b(self):
        """Return the bias."""
        return self.__b

    @property
    def A(self):
        """Return the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculate and return the sigmoid activation."""
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        return -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Return predictions and their cost."""
        A = self.forward_prop(X)
        return (np.where(A >= 0.5, 1, 0), self.cost(Y, A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one gradient descent update."""
        m = Y.shape[1]
        dz = A - Y
        self.__W -= alpha * np.matmul(dz, X.T) / m
        self.__b -= alpha * np.sum(dz) / m
