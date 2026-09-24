#!/usr/bin/env python3
"""Calculate bias-corrected exponential moving averages."""


def moving_average(data, beta):
    """Return the bias-corrected moving average of data."""
    averages = []
    current = 0
    for index, value in enumerate(data):
        current = beta * current + (1 - beta) * value
        correction = 1 - beta ** (index + 1)
        averages.append(current / correction)
    return averages
