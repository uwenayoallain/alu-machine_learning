#!/usr/bin/env python3
"""Update a neural network with L2-regularized gradient descent."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update network weights and biases in place using L2 regularization."""
    m = Y.shape[1]
    gradients = {}
    dZ_next = None

    for layer in range(L, 0, -1):
        activation = cache['A' + str(layer)]
        if layer == L:
            dZ = activation - Y
        else:
            dZ = np.matmul(
                weights['W' + str(layer + 1)].T, dZ_next)
            dZ *= 1 - activation ** 2
        dW = np.matmul(dZ, cache['A' + str(layer - 1)].T) / m
        dW += (lambtha / m) * weights['W' + str(layer)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        gradients['W' + str(layer)] = dW
        gradients['b' + str(layer)] = db
        dZ_next = dZ

    for layer in range(1, L + 1):
        weights['W' + str(layer)] -= alpha * gradients['W' + str(layer)]
        weights['b' + str(layer)] -= alpha * gradients['b' + str(layer)]
