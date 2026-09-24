#!/usr/bin/env python3
"""Implement one bias-corrected Adam optimization update."""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update a variable and both Adam moment estimates."""
    first_moment = beta1 * v + (1 - beta1) * grad
    second_moment = beta2 * s + (1 - beta2) * grad ** 2
    corrected_first = first_moment / (1 - beta1 ** t)
    corrected_second = second_moment / (1 - beta2 ** t)
    updated = var - alpha * corrected_first / (
        np.sqrt(corrected_second) + epsilon)
    return updated, first_moment, second_moment
