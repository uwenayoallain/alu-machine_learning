#!/usr/bin/env python3
"""Adds evaluation to the neural network."""

import numpy as np
from importlib import import_module

_Base = import_module('11-neural_network').NeuralNetwork


class NeuralNetwork(_Base):
    """A neural network that returns binary predictions."""

    def evaluate(self, X, Y):
        """Return predictions and cost."""
        self.forward_prop(X)
        return np.where(self.A2 >= 0.5, 1, 0), self.cost(Y, self.A2)
