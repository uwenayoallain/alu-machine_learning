#!/usr/bin/env python3
"""Adds training to the neural network."""

import numpy as np


class NeuralNetwork:
    """A trainable one-hidden-layer binary classifier."""

    def __init__(self, nx, nodes):
        """Initialize the network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Return first-layer weights."""
        return self.__W1

    @property
    def b1(self):
        """Return first-layer bias."""
        return self.__b1

    @property
    def A1(self):
        """Return first-layer activation."""
        return self.__A1

    @property
    def W2(self):
        """Return second-layer weights."""
        return self.__W2

    @property
    def b2(self):
        """Return second-layer bias."""
        return self.__b2

    @property
    def A2(self):
        """Return second-layer activation."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate and return both layer activations."""
        self.__A1 = 1 / (1 + np.exp(-(
            np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(
            np.matmul(self.__W2, self.__A1) + self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        return -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Return predictions and cost."""
        _, A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Train and evaluate the network."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
