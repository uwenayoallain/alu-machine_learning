#!/usr/bin/env python3
"""Calculate the integral of a polynomial."""


def poly_integral(poly, C=0):
    """Return the coefficient list for the integral of a polynomial."""
    if (type(poly) is not list or not poly or type(C) is not int or
            not all(type(coef) in (int, float) for coef in poly)):
        return None

    integral = [C]
    for power, coefficient in enumerate(poly, 1):
        value = coefficient / power
        integral.append(int(value) if value.is_integer() else value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
