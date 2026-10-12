#!/usr/bin/env python3
"""Calculate precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """Return true positives divided by predicted positives per class."""
    return np.diag(confusion) / np.sum(confusion, axis=0)
