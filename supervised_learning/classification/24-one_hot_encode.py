#!/usr/bin/env python3
"""Provides one-hot encoding."""

import numpy as np


def one_hot_encode(Y, classes):
    """Convert labels to a classes-by-examples one-hot matrix."""
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    try:
        encoded = np.zeros((classes, Y.shape[0]))
        encoded[Y, np.arange(Y.shape[0])] = 1
        return encoded
    except (IndexError, ValueError):
        return None
