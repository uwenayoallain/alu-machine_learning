#!/usr/bin/env python3
"""Adds logistic cost to the neural network."""

import numpy as np
from importlib import import_module

_Base = import_module('10-neural_network').NeuralNetwork


class NeuralNetwork(_Base):
    """A sigmoid network with logistic cost."""

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        return -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / Y.shape[1]
