#!/usr/bin/env python3
"""Calculate specificity for each class in a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Return true negatives divided by actual negatives for each class."""
    actual = np.sum(confusion, axis=1)
    predicted = np.sum(confusion, axis=0)
    negatives = np.sum(confusion) - actual
    true_negatives = negatives - predicted + np.diag(confusion)
    return true_negatives / negatives
