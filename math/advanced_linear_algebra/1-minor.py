#!/usr/bin/env python3
"""Calculate the minor matrix of a matrix."""


def determinant(matrix):
    """Return the determinant of a square matrix."""
    if matrix == []:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    total = 0
    for column, value in enumerate(matrix[0]):
        submatrix = [row[:column] + row[column + 1:] for row in matrix[1:]]
        total += (-1) ** column * value * determinant(submatrix)
    return total


def minor(matrix):
    """Return the matrix containing every element's minor."""
    if not isinstance(matrix, list) or matrix == [] or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    return [[determinant([row[:j] + row[j + 1:] for row in
                          matrix[:i] + matrix[i + 1:]])
             for j in range(len(matrix))] for i in range(len(matrix))]
