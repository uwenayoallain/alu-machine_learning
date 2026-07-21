#!/usr/bin/env python3
"""Perform convolution on images containing channels."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolve multi-channel images with one kernel."""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
        pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
        oh, ow = h, w
    else:
        if padding == 'valid':
            ph, pw = 0, 0
        else:
            ph, pw = padding
        pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
        oh = (h + (2 * ph) - kh) // sh + 1
        ow = (w + (2 * pw) - kw) // sw + 1
    padded = np.pad(images, pad_width, mode='constant')
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            row = i * sh
            column = j * sw
            region = padded[:, row:row + kh, column:column + kw, :]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))
    return output
