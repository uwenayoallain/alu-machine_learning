#!/usr/bin/env python3
"""Calculate per-class F1 scores using precision and sensitivity."""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Return the harmonic mean of precision and recall for each class."""
    recall = sensitivity(confusion)
    positive_precision = precision(confusion)
    return 2 * positive_precision * recall / (positive_precision + recall)
