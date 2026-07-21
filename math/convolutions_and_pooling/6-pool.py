#!/usr/bin/env python3
"""Perform pooling on images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Perform max or average pooling on multi-channel images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    output = np.zeros((m, oh, ow, c))
    for i in range(oh):
        for j in range(ow):
            row = i * sh
            column = j * sw
            region = images[:, row:row + kh, column:column + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
    return output
