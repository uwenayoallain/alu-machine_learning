#!/usr/bin/env python3
"""Defines a private deep binary classification network."""

import numpy as np


class DeepNeuralNetwork:
    """A deep network with read-only properties."""

    def __init__(self, nx, layers):
        """Initialize He weights and zero biases."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if not layers or not all(isinstance(n, int) and n > 0
                                  for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        previous = nx
        for layer, nodes in enumerate(layers, 1):
            self.__weights['W{}'.format(layer)] = (np.random.randn(nodes,
                previous) * np.sqrt(2 / previous))
            self.__weights['b{}'.format(layer)] = np.zeros((nodes, 1))
            previous = nodes

    @property
    def L(self):
        """Return the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Return the activation cache."""
        return self.__cache

    @property
    def weights(self):
        """Return the weights dictionary."""
        return self.__weights
