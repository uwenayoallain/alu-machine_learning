#!/usr/bin/env python3
"""Adds evaluation to a deep network."""

import numpy as np
from importlib import import_module

_Base = import_module('19-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(_Base):
    """A deep network that returns binary predictions."""

    def evaluate(self, X, Y):
        """Return predictions and cost."""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)
