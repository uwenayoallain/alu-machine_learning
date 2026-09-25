#!/usr/bin/env python3
"""Implement Bayesian optimization with Expected Improvement."""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Perform Bayesian optimization over a one-dimensional domain."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the objective, Gaussian process, and sample grid."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Return the next point and Expected Improvement values."""
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1)
        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        z = np.zeros_like(sigma)
        positive = sigma > 0
        z[positive] = improvement[positive] / sigma[positive]
        expected_improvement = np.zeros_like(sigma)
        values = improvement[positive] * norm.cdf(z[positive])
        values += sigma[positive] * norm.pdf(z[positive])
        expected_improvement[positive] = values
        X_next = self.X_s[np.argmax(expected_improvement)]
        return X_next, expected_improvement

    def optimize(self, iterations=100):
        """Optimize the objective and return the best sampled point."""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if np.any(np.isclose(self.gp.X, X_next).all(axis=1)):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        return self.gp.X[index].copy(), self.gp.Y[index].copy()
