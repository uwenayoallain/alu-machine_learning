#!/usr/bin/env python3
"""Calculate the derivative of a polynomial."""


def poly_derivative(poly):
    """Return the coefficient list for the derivative of a polynomial."""
    if (type(poly) is not list or not poly or
            not all(type(coef) in (int, float) for coef in poly)):
        return None
    if len(poly) == 1:
        return [0]

    derivative = [power * poly[power] for power in range(1, len(poly))]
    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()
    return derivative
