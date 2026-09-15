#!/usr/bin/env python3
"""Adds selectable hidden-layer activations to the deep network."""

import numpy as np
from importlib import import_module

_Base = import_module('27-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(_Base):
    """A multiclass deep network using sigmoid or tanh hidden layers."""

    def __init__(self, nx, layers, activation='sig'):
        """Initialize the network and select its hidden activation."""
        super().__init__(nx, layers)
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation

    @property
    def activation(self):
        """Return the hidden-layer activation name."""
        return self.__activation

    def forward_prop(self, X):
        """Calculate activations using the selected hidden function."""
        cache = self.cache
        cache['A0'] = X
        for layer in range(1, self.L + 1):
            previous = cache['A{}'.format(layer - 1)]
            z = np.matmul(self.weights['W{}'.format(layer)], previous)
            z += self.weights['b{}'.format(layer)]
            if layer == self.L:
                exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
                cache['A{}'.format(layer)] = exp_z / np.sum(
                    exp_z, axis=0, keepdims=True)
            elif self.__activation == 'sig':
                cache['A{}'.format(layer)] = 1 / (1 + np.exp(-z))
            else:
                cache['A{}'.format(layer)] = np.tanh(z)
        return cache['A{}'.format(self.L)], cache

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Backpropagate through the selected hidden activation."""
        m = Y.shape[1]
        dz = cache['A{}'.format(self.L)] - Y
        for layer in range(self.L, 0, -1):
            previous = cache['A{}'.format(layer - 1)]
            dw = np.matmul(dz, previous.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.weights['W{}'.format(layer)] -= alpha * dw
            self.weights['b{}'.format(layer)] -= alpha * db
            if layer > 1:
                dz = np.matmul(self.weights['W{}'.format(layer)].T, dz)
                hidden = cache['A{}'.format(layer - 1)]
                if self.__activation == 'sig':
                    dz *= hidden * (1 - hidden)
                else:
                    dz *= 1 - hidden ** 2
