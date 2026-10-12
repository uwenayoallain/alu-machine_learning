# Error Analysis

Confusion matrices use rows for actual classes and columns for predictions.

- `0-create_confusion.py`: count actual/predicted class pairs.
- `1-sensitivity.py`: recall, TP / (TP + FN).
- `2-precision.py`: precision, TP / (TP + FP).
- `3-specificity.py`: specificity, TN / (TN + FP).
- `4-f1_score.py`: harmonic mean of precision and recall.
- `5-error_handling`: approaches to bias and variance scenarios.

Python modules target Python 3.5 and NumPy 1.15. Undefined ratios retain
NumPy's division-by-zero behavior. The numbered theory answers are plain text.
