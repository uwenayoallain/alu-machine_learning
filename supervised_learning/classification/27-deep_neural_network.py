#!/usr/bin/env python3
"""Updates the deep network for multiclass classification."""

import numpy as np
_namespace = {}
exec(open('26-deep_neural_network.py').read(), _namespace)
_Base = _namespace['DeepNeuralNetwork']


class DeepNeuralNetwork(_Base):
    """A deep network with a softmax output layer."""

    def forward_prop(self, X):
        """Calculate hidden sigmoid activations and softmax output."""
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
            else:
                cache['A{}'.format(layer)] = 1 / (1 + np.exp(-z))
        return cache['A{}'.format(self.L)], cache

    def cost(self, Y, A):
        """Calculate multiclass cross-entropy cost."""
        return -np.sum(Y * np.log(A + 1.0000001e-7)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Return one-hot predictions and multiclass cost."""
        A, _ = self.forward_prop(X)
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, self.cost(Y, A)
