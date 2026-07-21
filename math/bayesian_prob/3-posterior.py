#!/usr/bin/env python3
"""Calculate posterior probabilities for binomial hypotheses."""

import numpy as np

intersection = __import__('1-intersection').intersection


def posterior(x, n, P, Pr):
    """Calculate the posterior probability of each hypothesis in P."""
    joint = intersection(x, n, P, Pr)
    return joint / np.sum(joint)
