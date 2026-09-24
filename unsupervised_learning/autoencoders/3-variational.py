#!/usr/bin/env python3
"""Create a dense variational autoencoder."""

import tensorflow.keras as keras


class VAEOutput(keras.layers.Layer):
    """Add the VAE reconstruction and KL losses to a model."""

    def __init__(self, input_dims, **kwargs):
        """Initialize the output layer."""
        super(VAEOutput, self).__init__(**kwargs)
        self.input_dims = input_dims

    def call(self, values):
        """Return reconstruction while registering the VAE loss."""
        inputs, reconstruction, mean, log_variance = values
        backend = keras.backend
        reconstruction_loss = backend.mean(
            backend.binary_crossentropy(inputs, reconstruction)
        )
        kl_loss = backend.mean(-0.5 * backend.sum(
            1 + log_variance - backend.square(mean)
            - backend.exp(log_variance),
            axis=-1
        ))
        self.add_loss(
            (self.input_dims - 1) * reconstruction_loss + kl_loss
        )
        return reconstruction


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
    mean = keras.layers.Dense(latent_dims, activation='linear')(encoded)
    log_variance = keras.layers.Dense(
        latent_dims,
        activation='linear'
    )(encoded)
    latent = keras.layers.Lambda(
        sampling,
        output_shape=(latent_dims,)
    )([mean, log_variance])
    encoder = keras.Model(
        encoder_input,
        [latent, mean, log_variance],
        name='encoder'
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input
    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(units, activation='relu')(decoded)
    output = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, output, name='decoder')

    auto_input = keras.Input(shape=(input_dims,))
    latent, mean, log_variance = encoder(auto_input)
    reconstruction = decoder(latent)
    reconstruction = VAEOutput(input_dims)(
        [auto_input, reconstruction, mean, log_variance]
    )
    auto = keras.Model(auto_input, reconstruction, name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
