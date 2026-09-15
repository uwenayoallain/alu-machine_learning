#!/usr/bin/env python3
"""Creates initialized TensorFlow dense layers."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Return a dense layer with variance-scaled initialization."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    return tf.layers.Dense(n, activation=activation,
                           kernel_initializer=initializer,
                           name='layer')(prev)
