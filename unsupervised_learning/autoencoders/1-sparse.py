#!/usr/bin/env python3
"""Create a sparse dense autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder with L1 latent regularization."""
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input
    for units in hidden_layers:
        encoded = keras.layers.Dense(units, activation='relu')(encoded)
    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)
    encoder = keras.Model(encoder_input, latent, name='encoder')

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(units, activation='relu')(decoded)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, output, name='decoder')

    auto_input = keras.Input(shape=(input_dims,))
    reconstruction = decoder(encoder(auto_input))
    auto = keras.Model(
        auto_input,
        reconstruction,
        name='sparse_autoencoder'
    )
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
