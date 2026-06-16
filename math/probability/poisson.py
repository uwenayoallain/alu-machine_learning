#!/usr/bin/env python3
"""Represent a Poisson probability distribution."""


class Poisson:
    """Represent a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initialize a Poisson distribution from data or a rate."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculate the probability of observing exactly k successes."""
        k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for value in range(1, k + 1):
            factorial *= value
        return (2.7182818285 ** -self.lambtha
                * self.lambtha ** k / factorial)

    def cdf(self, k):
        """Calculate the probability of observing at most k successes."""
        k = int(k)
        if k < 0:
            return 0

        total = 0
        factorial = 1
        for value in range(k + 1):
            if value > 0:
                factorial *= value
            total += self.lambtha ** value / factorial
        return 2.7182818285 ** -self.lambtha * total
