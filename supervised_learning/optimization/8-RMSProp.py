#!/usr/bin/env python3
"""Create a TensorFlow RMSProp optimization operation."""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Return an operation that minimizes loss with RMSProp."""
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, momentum=0.0, epsilon=epsilon)
    return optimizer.minimize(loss)
