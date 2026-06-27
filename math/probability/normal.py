#!/usr/bin/env python3
"""Represent a normal probability distribution."""


class Normal:
    """Represent a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a normal distribution from data or its parameters."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((value - self.mean) ** 2 for value in data)
            self.stddev = (variance / len(data)) ** 0.5

    def z_score(self, x):
        """Calculate the z-score corresponding to x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x-value corresponding to a z-score."""
        return self.mean + z * self.stddev

    def pdf(self, x):
        """Calculate the probability density at x."""
        coefficient = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        return coefficient * 2.7182818285 ** (-0.5 * self.z_score(x) ** 2)

    def cdf(self, x):
        """Calculate the probability of observing a value at most x."""
        value = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = value - value ** 3 / 3 + value ** 5 / 10
        erf -= value ** 7 / 42 - value ** 9 / 216
        erf *= 2 / 3.14159265359 ** 0.5
        return (1 + erf) / 2
