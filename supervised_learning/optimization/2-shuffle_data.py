#!/usr/bin/env python3
"""Shuffle paired data matrices with one shared permutation."""

import numpy as np


def shuffle_data(X, Y):
    """Return X and Y shuffled with the same row permutation."""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
