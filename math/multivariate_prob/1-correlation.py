#!/usr/bin/env python3
"""Calculate a correlation matrix from a covariance matrix."""

import numpy as np


def correlation(C):
    """Return the correlation matrix represented by covariance matrix C."""
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    standard_deviations = np.sqrt(np.diag(C))
    return C / np.outer(standard_deviations, standard_deviations)
