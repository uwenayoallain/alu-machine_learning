#!/usr/bin/env python3
"""Determine the definiteness of a matrix."""
import numpy as np


def definiteness(matrix):
    """Return the definiteness classification of a symmetric matrix."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not matrix.size):
        return None
    if not np.array_equal(matrix, matrix.T):
        return None
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = np.finfo(float).eps * max(1, matrix.shape[0])
    tolerance *= max(1, np.max(np.abs(eigenvalues)))
    if np.all(eigenvalues > tolerance):
        return "Positive definite"
    if np.all(eigenvalues < -tolerance):
        return "Negative definite"
    if np.all(eigenvalues >= -tolerance):
        return "Positive semi-definite"
    if np.all(eigenvalues <= tolerance):
        return "Negative semi-definite"
    return "Indefinite"
