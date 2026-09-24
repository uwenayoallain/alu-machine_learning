#!/usr/bin/env python3
"""Calculate staircase inverse-time learning-rate decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return the decayed learning rate for a global step."""
    return alpha / (1 + decay_rate * (global_step // decay_step))
