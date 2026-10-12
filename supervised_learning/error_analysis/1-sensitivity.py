#!/usr/bin/env python3
"""Calculate recall for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Return true positives divided by actual positives for each class."""
    return np.diag(confusion) / np.sum(confusion, axis=1)
