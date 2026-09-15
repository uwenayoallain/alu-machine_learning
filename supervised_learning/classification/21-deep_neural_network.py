#!/usr/bin/env python3
"""Adds backpropagation to a deep network."""

import numpy as np
_namespace = {}
exec(open('20-deep_neural_network.py').read(), _namespace)
_Base = _namespace['DeepNeuralNetwork']


class DeepNeuralNetwork(_Base):
    """A trainable deep sigmoid network."""

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one backward pass and update every layer."""
        m = Y.shape[1]
        dz = cache['A{}'.format(self.L)] - Y
        for layer in range(self.L, 0, -1):
            current = cache['A{}'.format(layer - 1)]
            weights = self.weights['W{}'.format(layer)].copy()
            dw = np.matmul(dz, current.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.weights['W{}'.format(layer)] -= alpha * dw
            self.weights['b{}'.format(layer)] -= alpha * db
            if layer > 1:
                dz = np.matmul(weights.T, dz)
                dz *= cache['A{}'.format(layer - 1)] * (
                    1 - cache['A{}'.format(layer - 1)])
