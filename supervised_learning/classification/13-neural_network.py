#!/usr/bin/env python3
"""Adds gradient descent to the neural network."""

import numpy as np
from importlib import import_module

_Base = import_module('12-neural_network').NeuralNetwork


class NeuralNetwork(_Base):
    """A trainable one-hidden-layer binary classifier."""

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Perform one backpropagation update."""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self._NeuralNetwork__W1 -= alpha * dw1
        self._NeuralNetwork__b1 -= alpha * db1
        self._NeuralNetwork__W2 -= alpha * dw2
        self._NeuralNetwork__b2 -= alpha * db2
