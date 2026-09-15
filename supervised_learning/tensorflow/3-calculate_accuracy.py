#!/usr/bin/env python3
"""Builds a TensorFlow accuracy tensor."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Return the mean fraction of correctly classified examples."""
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
