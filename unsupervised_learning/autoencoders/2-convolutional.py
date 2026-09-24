#!/usr/bin/env python3
"""Create a convolutional autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder and its encoder and decoder."""
    encoder_input = keras.Input(shape=input_dims)
    encoded = encoder_input
    for filter_count in filters:
        encoded = keras.layers.Conv2D(
            filter_count,
            (3, 3),
            padding='same',
            activation='relu'
        )(encoded)
        encoded = keras.layers.MaxPooling2D(
            (2, 2),
            padding='same'
        )(encoded)
    encoder = keras.Model(encoder_input, encoded, name='encoder')

    decoder_input = keras.Input(shape=latent_dims)
    decoded = decoder_input
    reversed_filters = list(reversed(filters))
    for index, filter_count in enumerate(reversed_filters):
        padding = 'valid' if index == len(reversed_filters) - 1 else 'same'
        decoded = keras.layers.Conv2D(
            filter_count,
            (3, 3),
            padding=padding,
            activation='relu'
        )(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    output = keras.layers.Conv2D(
        input_dims[2],
        (3, 3),
        padding='same',
        activation='sigmoid'
    )(decoded)
    decoder = keras.Model(decoder_input, output, name='decoder')

    auto_input = keras.Input(shape=input_dims)
    reconstruction = decoder(encoder(auto_input))
    auto = keras.Model(auto_input, reconstruction, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
