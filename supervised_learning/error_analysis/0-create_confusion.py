#!/usr/bin/env python3
"""Create a confusion matrix from one-hot labels and predictions."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Return counts with actual classes as rows and predictions as columns."""
    confusion = np.zeros((labels.shape[1], labels.shape[1]))
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)
    np.add.at(confusion, (actual, predicted), 1)
    return confusion
