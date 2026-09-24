#!/usr/bin/env python3
"""Implement one RMSProp optimization update."""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Update a variable and its second moment with RMSProp."""
    second_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated = var - alpha * grad / (np.sqrt(second_moment) + epsilon)
    return updated, second_moment
