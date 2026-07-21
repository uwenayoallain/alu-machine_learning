#!/usr/bin/env python3
"""Perform same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolve grayscale images while preserving their dimensions."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph_before = kh // 2
    ph_after = kh - 1 - ph_before
    pw_before = kw // 2
    pw_after = kw - 1 - pw_before
    padded = np.pad(images,
                    ((0, 0), (ph_before, ph_after),
                     (pw_before, pw_after)),
                    mode='constant')
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    return output
