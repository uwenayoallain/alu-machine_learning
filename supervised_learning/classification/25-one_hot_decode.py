#!/usr/bin/env python3
"""Provides one-hot decoding."""

import numpy as np


def one_hot_decode(one_hot):
    """Convert a one-hot matrix to a label vector."""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    return np.argmax(one_hot, axis=0)
