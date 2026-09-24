#!/usr/bin/env python3
"""Create a TensorFlow inverse-time learning-rate decay operation."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return a staircase inverse-time learning-rate tensor."""
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
