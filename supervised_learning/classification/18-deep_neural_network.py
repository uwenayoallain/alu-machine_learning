#!/usr/bin/env python3
"""Adds forward propagation to a deep network."""

import numpy as np
_namespace = {}
exec(open('17-deep_neural_network.py').read(), _namespace)
_Base = _namespace['DeepNeuralNetwork']


class DeepNeuralNetwork(_Base):
    """A deep sigmoid network."""

    def forward_prop(self, X):
        """Calculate all activations and return output and cache."""
        cache = self.cache
        cache['A0'] = X
        for layer in range(1, self.L + 1):
            previous = cache['A{}'.format(layer - 1)]
            z = np.matmul(self.weights['W{}'.format(layer)], previous)
            z = z + self.weights['b{}'.format(layer)]
            cache['A{}'.format(layer)] = 1 / (1 + np.exp(-z))
        return cache['A{}'.format(self.L)], cache
