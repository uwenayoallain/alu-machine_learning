#!/usr/bin/env python3
"""Calculate a sum of consecutive squares."""


def summation_i_squared(n):
    """Return the sum of the squares of all integers from 1 through n."""
    if type(n) is not int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
