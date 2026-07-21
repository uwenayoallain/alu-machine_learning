#!/usr/bin/env python3
"""Calculate the marginal probability of binomial data."""

import numpy as np

intersection = __import__('1-intersection').intersection


def marginal(x, n, P, Pr):
    """Calculate the marginal probability of observing x in n trials."""
    return np.sum(intersection(x, n, P, Pr))
