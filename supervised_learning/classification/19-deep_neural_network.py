#!/usr/bin/env python3
"""Adds logistic cost to a deep network."""

import numpy as np
_namespace = {}
exec(open('18-deep_neural_network.py').read(), _namespace)
_Base = _namespace['DeepNeuralNetwork']


class DeepNeuralNetwork(_Base):
    """A deep sigmoid network with logistic cost."""

    def cost(self, Y, A):
        """Calculate logistic regression cost."""
        return -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / Y.shape[1]
