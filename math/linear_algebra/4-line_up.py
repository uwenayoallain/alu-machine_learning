#!/usr/bin/env python3
"""Add two arrays element by element."""


def add_arrays(arr1, arr2):
    """Return the element-wise sum of equally sized arrays."""
    if len(arr1) != len(arr2):
        return None
    return [arr1[index] + arr2[index] for index in range(len(arr1))]
