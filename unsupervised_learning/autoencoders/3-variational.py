#!/usr/bin/env python3
"""Create a dense variational autoencoder."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder and its encoder and decoder."""
    backend = keras.backend

    def sampling(arguments):
        """Sample a latent vector using the reparameterization trick."""
        mean, log_variance = arguments
        noise = backend.random_normal(shape=backend.shape(mean))
        return mean + backend.exp(0.5 * log_variance) * noise

    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input
    for units in hidden_layers:
        encoded = keras.layers.Dense(units, activation='relu')(encoded)
    mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    log_variance = keras.layers.Dense(latent_dims, activation=None)(encoded)
    latent = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([mean, log_variance])
    encoder = keras.Model(
        encoder_input,
        [latent, mean, log_variance]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(units, activation='relu')(decoded)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, output)

    auto_input = keras.Input(shape=(input_dims,))
    latent, mean, log_variance = encoder(auto_input)
    reconstruction = decoder(latent)
    auto = keras.Model(auto_input, reconstruction)

    reconstruction_loss = keras.losses.binary_crossentropy(
        auto_input,
        reconstruction
    )
    reconstruction_loss *= input_dims
    kl_loss = -0.5 * backend.sum(
        1 + log_variance - backend.square(mean)
        - backend.exp(log_variance),
        axis=-1
    )
    auto.add_loss(backend.mean(reconstruction_loss + kl_loss))
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
