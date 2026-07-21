#!/usr/bin/env python3
"""Concatenate two-dimensional matrices."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Return matrices concatenated along axis zero or one."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[index] + mat2[index] for index in range(len(mat1))]
    return None
