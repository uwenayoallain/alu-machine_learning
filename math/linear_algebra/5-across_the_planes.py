#!/usr/bin/env python3
"""Add two two-dimensional matrices."""


def add_matrices2D(mat1, mat2):
    """Return the element-wise sum of equally shaped matrices."""
    if len(mat1) != len(mat2):
        return None
    if len(mat1) > 0 and len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[row][column] + mat2[row][column]
             for column in range(len(mat1[row]))]
            for row in range(len(mat1))]
