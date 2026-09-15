#!/usr/bin/env python3
"""Loads and evaluates a TensorFlow classifier checkpoint."""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Return predictions, accuracy, and loss from a saved model."""
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(session, save_path)
        graph = tf.get_default_graph()
        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        y_pred = graph.get_collection('y_pred')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]
        return session.run((y_pred, accuracy, loss),
                          feed_dict={x: X, y: Y})
