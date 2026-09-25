#!/usr/bin/env python3
"""Create a TensorFlow dense layer followed by dropout."""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Return the output of a variance-scaled dense layer with dropout."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)
    output = layer(prev)
    if keep_prob == 0:
        return output
    dropout = tf.layers.Dropout(rate=1 - keep_prob)
    return dropout(output)
