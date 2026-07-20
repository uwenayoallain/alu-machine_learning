#!/usr/bin/env python3
"""Calculate likelihoods for a binomial experiment."""

import numpy as np


def likelihood(x, n, P):
    """Calculate the likelihood of observing x successes in n trials."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    coefficient = 1
    for i in range(1, x + 1):
        coefficient *= (n - i + 1) / i
    return coefficient * (P ** x) * ((1 - P) ** (n - x))
