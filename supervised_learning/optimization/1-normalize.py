#!/usr/bin/env python3
"""Normalize a matrix using feature means and standard deviations."""

import numpy as np


def normalize(X, m, s):
    """Return the standardized feature matrix."""
    return (X - m) / s
