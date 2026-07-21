#!/usr/bin/env python3
"""Calculate the determinant of a matrix."""


def determinant(matrix):
    """Return the determinant of a square matrix."""
    if not isinstance(matrix, list) or matrix == [] or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    result = 0
    for column, value in enumerate(matrix[0]):
        submatrix = [row[:column] + row[column + 1:] for row in matrix[1:]]
        result += ((-1) ** column) * value * determinant(submatrix)
    return result
