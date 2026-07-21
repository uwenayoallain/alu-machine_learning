#!/usr/bin/env python3
"""Convolve images using multiple kernels."""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Convolve multi-channel images with multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        oh = int(np.ceil(h / sh))
        ow = int(np.ceil(w / sw))
        total_h = max((oh - 1) * sh + kh - h, 0)
        total_w = max((ow - 1) * sw + kw - w, 0)
        ph_before = total_h // 2
        pw_before = total_w // 2
        pad_width = ((0, 0), (ph_before, total_h - ph_before),
                     (pw_before, total_w - pw_before), (0, 0))
    else:
        if padding == 'valid':
            ph, pw = 0, 0
        else:
            ph, pw = padding
        pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
        oh = (h + (2 * ph) - kh) // sh + 1
        ow = (w + (2 * pw) - kw) // sw + 1
    padded = np.pad(images, pad_width, mode='constant')
    output = np.zeros((m, oh, ow, nc))
    for i in range(oh):
        for j in range(ow):
            row = i * sh
            column = j * sw
            region = padded[:, row:row + kh, column:column + kw, :]
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    region * kernels[:, :, :, k], axis=(1, 2, 3))
    return output
