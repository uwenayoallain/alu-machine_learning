#!/usr/bin/env python3
"""Multiply two two-dimensional matrices."""


def mat_mul(mat1, mat2):
    """Return the matrix product of compatible matrices."""
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[row][item] * mat2[item][column]
                 for item in range(len(mat2)))
             for column in range(len(mat2[0]))]
            for row in range(len(mat1))]
