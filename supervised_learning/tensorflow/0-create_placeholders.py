#!/usr/bin/env python3
"""Creates TensorFlow placeholders for classifier data."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Return input and one-hot label placeholders."""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
