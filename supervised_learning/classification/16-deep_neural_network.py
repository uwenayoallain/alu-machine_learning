#!/usr/bin/env python3
"""Defines a deep binary classification network."""

import numpy as np


class DeepNeuralNetwork:
    """A fully connected deep neural network."""

    def __init__(self, nx, layers):
        """Initialize He weights and zero biases."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        previous = nx
        for layer, nodes in enumerate(layers, 1):
            self.weights['W{}'.format(layer)] = (np.random.randn(nodes,
                previous) * np.sqrt(2 / previous))
            self.weights['b{}'.format(layer)] = np.zeros((nodes, 1))
            previous = nodes
