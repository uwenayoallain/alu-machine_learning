#!/usr/bin/env python3
"""Calculate the adjugate matrix of a matrix."""


def determinant(matrix):
    """Return the determinant of a square matrix."""
    if matrix == []:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    return sum((-1) ** j * value * determinant(
        [row[:j] + row[j + 1:] for row in matrix[1:]])
        for j, value in enumerate(matrix[0]))


def adjugate(matrix):
    """Return the transpose of a matrix's cofactor matrix."""
    if not isinstance(matrix, list) or matrix == [] or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    cofactors = [[(-1) ** (i + j) * determinant(
        [row[:j] + row[j + 1:] for row in matrix[:i] + matrix[i + 1:]])
        for j in range(len(matrix))] for i in range(len(matrix))]
    return [[cofactors[j][i] for j in range(len(matrix))]
            for i in range(len(matrix))]
