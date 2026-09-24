#!/usr/bin/env python3
"""Implement batch normalization for a NumPy feature matrix."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalize, scale, and shift each feature column."""
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta
