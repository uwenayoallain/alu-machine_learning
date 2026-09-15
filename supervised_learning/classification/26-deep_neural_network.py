#!/usr/bin/env python3
"""Adds persistence to a deep neural network."""

import pickle
from importlib import import_module

_Base = import_module('23-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(_Base):
    """A persistent deep binary classifier."""

    def save(self, filename):
        """Save this network to a pickle file."""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """Load a network from pickle, or return None if absent."""
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
