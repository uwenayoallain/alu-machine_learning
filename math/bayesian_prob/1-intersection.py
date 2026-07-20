#!/usr/bin/env python3
"""Calculate joint probabilities for a binomial experiment."""

import numpy as np

likelihood = __import__('0-likelihood').likelihood


def intersection(x, n, P, Pr):
    """Calculate intersections of observed data and hypotheses."""
    probabilities = likelihood(x, n, P)
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    return probabilities * Pr
