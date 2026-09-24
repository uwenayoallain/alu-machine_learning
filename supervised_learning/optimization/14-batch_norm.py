#!/usr/bin/env python3
"""Create a TensorFlow batch-normalized dense layer."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Return the activated output of a normalized dense layer."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, kernel_initializer=initializer, activation=None)
    Z = layer(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), name="gamma")
    beta = tf.Variable(tf.zeros([n]), name="beta")
    normalized = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, 1e-8)
    if activation is None:
        return normalized
    return activation(normalized)
