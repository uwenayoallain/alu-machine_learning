#!/usr/bin/env python3
"""Add TensorFlow regularization losses to a cost tensor."""

import tensorflow as tf


def l2_reg_cost(cost):
    """Return cost plus the collected TensorFlow regularization losses."""
    return cost + tf.losses.get_regularization_loss()
