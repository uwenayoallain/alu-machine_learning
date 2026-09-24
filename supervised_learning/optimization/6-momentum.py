#!/usr/bin/env python3
"""Create a TensorFlow momentum optimization operation."""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Return an operation that minimizes loss with momentum."""
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=alpha, momentum=beta1)
    return optimizer.minimize(loss)
