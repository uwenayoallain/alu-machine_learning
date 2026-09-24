#!/usr/bin/env python3
"""Calculate normalization constants for feature columns."""

import numpy as np


def normalization_constants(X):
    """Return the mean and standard deviation of each feature."""
    return np.mean(X, axis=0), np.std(X, axis=0)
