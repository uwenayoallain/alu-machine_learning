#!/usr/bin/env python3
"""Build and train a batch-normalized TensorFlow classifier."""

import numpy as np
import tensorflow as tf


def _create_dense(prev, units, activation):
    """Return the output of a variance-scaled dense layer."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=units,
        activation=None,
        kernel_initializer=initializer)
    output = layer(prev)
    if activation is None:
        return output
    return activation(output)


def _create_batch_norm_layer(prev, units, activation):
    """Return the activated output of a batch-normalized dense layer."""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=units,
        activation=None,
        kernel_initializer=initializer)
    Z = layer(prev)
    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([units]), name="gamma")
    beta = tf.Variable(tf.zeros([units]), name="beta")
    normalized = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, 1e-8)
    if activation is None:
        return normalized
    return activation(normalized)


def _shuffle_data(X, Y):
    """Return paired data matrices shuffled with one permutation."""
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]


def _forward_prop(x, layers, activations):
    """Build the classifier output with batch-normalized hidden layers."""
    output = x
    last = len(layers) - 1
    for index, units in enumerate(layers):
        if index == last:
            output = _create_dense(output, units, activations[index])
        else:
            output = _create_batch_norm_layer(
                output, units, activations[index])
    return output


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """Build, train, and save the optimized classifier."""
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    examples = X_train.shape[0]
    features = X_train.shape[1]
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=(None, features), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = _forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y, logits=y_pred)
    tf.add_to_collection('loss', loss)

    global_step = tf.Variable(
        0, trainable=False, dtype=tf.int32, name='global_step')
    learning_rate = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=1,
        decay_rate=decay_rate,
        staircase=True)
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon).minimize(loss)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    initializer = tf.global_variables_initializer()
    batches = (examples + batch_size - 1) // batch_size

    with tf.Session() as session:
        session.run(initializer)
        for epoch in range(epochs + 1):
            session.run(global_step.assign(epoch))
            train_cost, train_accuracy = session.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            valid_cost, valid_accuracy = session.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch == epochs:
                continue

            shuffled_X, shuffled_Y = _shuffle_data(X_train, Y_train)
            for batch in range(batches):
                start = batch * batch_size
                stop = min(start + batch_size, examples)
                X_batch = shuffled_X[start:stop]
                Y_batch = shuffled_Y[start:stop]
                feed_dict = {x: X_batch, y: Y_batch}
                session.run(train_op, feed_dict=feed_dict)

                if (batch + 1) % 100 == 0:
                    step_cost, step_accuracy = session.run(
                        [loss, accuracy], feed_dict=feed_dict)
                    print("\tStep {}:".format(batch + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(session, save_path)
