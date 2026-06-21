#!/usr/bin/env python3
"""Represent an exponential probability distribution."""


class Exponential:
    """Represent an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize an exponential distribution from data or a rate."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            total = 0
            for value in data:
                total += value
            self.lambtha = float(1 / (total / len(data)))

    def pdf(self, x):
        """Calculate the probability density at time x."""
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """Calculate the probability of observing a time at most x."""
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
