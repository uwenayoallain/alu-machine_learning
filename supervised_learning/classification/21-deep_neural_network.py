#!/usr/bin/env python3
"""Adds backpropagation to a deep network."""

import numpy as np
from importlib import import_module

_Base = import_module('20-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(_Base):
    """A trainable deep sigmoid network."""

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one backward pass and update every layer."""
        m = Y.shape[1]
        dz = cache['A{}'.format(self.L)] - Y
        for layer in range(self.L, 0, -1):
            current = cache['A{}'.format(layer - 1)]
            dw = np.matmul(dz, current.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.weights['W{}'.format(layer)] -= alpha * dw
            self.weights['b{}'.format(layer)] -= alpha * db
            if layer > 1:
                dz = np.matmul(self.weights['W{}'.format(layer)].T, dz)
                dz *= cache['A{}'.format(layer - 1)] * (
                    1 - cache['A{}'.format(layer - 1)])
