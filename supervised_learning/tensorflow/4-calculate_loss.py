#!/usr/bin/env python3
"""Builds a TensorFlow softmax cross-entropy loss tensor."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Return the softmax cross-entropy loss."""
    return tf.losses.softmax_cross_entropy(y, y_pred)
