#!/usr/bin/env python3
"""Implement patience-based early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Return whether to stop and the updated non-improving count."""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
