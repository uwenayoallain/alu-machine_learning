#!/usr/bin/env python3
"""Builds a TensorFlow gradient-descent training operation."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Return a gradient-descent operation minimizing loss."""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
