#!/usr/bin/env python3
"""Builds a TensorFlow forward-propagation graph."""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Build and return the network prediction tensor."""
    layer = x
    for size, activation in zip(layer_sizes, activations):
        layer = create_layer(layer, size, activation)
    return layer
