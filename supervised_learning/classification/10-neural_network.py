#!/usr/bin/env python3
"""Adds forward propagation to the private neural network."""

import numpy as np
from importlib import import_module

_Base = import_module('9-neural_network').NeuralNetwork


class NeuralNetwork(_Base):
    """A sigmoid network with one hidden layer."""

    def forward_prop(self, X):
        """Calculate and return both layer activations."""
        self._NeuralNetwork__A1 = 1 / (1 + np.exp(-(
            np.matmul(self.W1, X) + self.b1)))
        self._NeuralNetwork__A2 = 1 / (1 + np.exp(-(
            np.matmul(self.W2, self.A1) + self.b2)))
        return self.A1, self.A2
