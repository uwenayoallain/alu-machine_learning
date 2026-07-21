#!/usr/bin/env python3
"""Calculate the shape of a matrix."""


def matrix_shape(matrix):
    """Return the dimensions of a nested list as a list of integers."""
    shape = []
    current = matrix
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape
