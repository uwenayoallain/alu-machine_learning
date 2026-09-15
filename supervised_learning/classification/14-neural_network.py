#!/usr/bin/env python3
"""Adds training to the neural network."""

from importlib import import_module

_Base = import_module('13-neural_network').NeuralNetwork


class NeuralNetwork(_Base):
    """A trainable one-hidden-layer binary classifier."""

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
