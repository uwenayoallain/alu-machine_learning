#!/usr/bin/env python3
"""Implement one gradient-descent update with momentum."""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update a variable and its first moment with momentum."""
    moment = beta1 * v + (1 - beta1) * grad
    updated = var - alpha * moment
    return updated, moment
