#!/usr/bin/env python3
"""Represent a binomial probability distribution."""


class Binomial:
    """Represent a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a binomial distribution from data or its parameters."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((value - mean) ** 2 for value in data)
            variance /= len(data)
            estimated_p = 1 - variance / mean
            self.n = round(mean / estimated_p)
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculate the probability of observing exactly k successes."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_factorial = 1
        k_factorial = 1
        difference_factorial = 1
        for value in range(1, self.n + 1):
            n_factorial *= value
        for value in range(1, k + 1):
            k_factorial *= value
        for value in range(1, self.n - k + 1):
            difference_factorial *= value
        coefficient = n_factorial / (k_factorial * difference_factorial)
        return (coefficient * self.p ** k
                * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculate the probability of observing at most k successes."""
        k = int(k)
        if k < 0:
            return 0
        total = 0
        for value in range(k + 1):
            total += self.pmf(value)
        return total
