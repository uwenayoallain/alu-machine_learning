#!/usr/bin/env python3
"""Calculate L2-regularized neural-network cost."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Return data cost plus the L2 penalty for all weight matrices."""
    norm = 0
    for layer in range(1, L + 1):
        norm += np.sum(weights['W' + str(layer)] ** 2)
    return cost + lambtha * norm / (2 * m)
