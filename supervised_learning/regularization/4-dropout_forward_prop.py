#!/usr/bin/env python3
"""Perform NumPy forward propagation with inverted dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Return layer activations and hidden-layer dropout masks."""
    cache = {'A0': X}
    for layer in range(1, L + 1):
        Z = np.matmul(
            weights['W' + str(layer)], cache['A' + str(layer - 1)])
        Z += weights['b' + str(layer)]
        activation = 'A' + str(layer)
        if layer == L:
            exp_Z = np.exp(Z)
            cache[activation] = exp_Z / np.sum(
                exp_Z, axis=0, keepdims=True)
        else:
            cache[activation] = np.tanh(Z)
            mask = np.random.binomial(
                1, keep_prob, size=cache[activation].shape)
            cache['D' + str(layer)] = mask
            cache[activation] *= mask / keep_prob
    return cache
