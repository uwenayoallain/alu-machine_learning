#!/usr/bin/env python3
"""Defines a trainable and observable sigmoid neuron."""

import matplotlib.pyplot as plt
import numpy as np


class Neuron:
    """Represents a binary classification neuron."""

    def __init__(self, nx):
        """Initialize the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Return weights."""
        return self.__W

    @property
    def b(self):
        """Return bias."""
        return self.__b

    @property
    def A(self):
        """Return activation."""
        return self.__A

    def forward_prop(self, X):
        """Run forward propagation."""
        self.__A = 1 / (1 + np.exp(-(np.matmul(self.__W, X) + self.__b)))
        return self.__A

    def cost(self, Y, A):
        """Calculate logistic cost."""
        return -np.sum(Y * np.log(A) + (1 - Y) *
                       np.log(1.0000001 - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Evaluate predictions and cost."""
        A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Perform one gradient descent update."""
        m = Y.shape[1]
        dz = A - Y
        self.__W -= alpha * np.matmul(dz, X.T) / m
        self.__b -= alpha * np.sum(dz) / m

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train the neuron, optionally printing and plotting costs."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        points = []
        if verbose or graph:
            costs.append(self.cost(Y, self.forward_prop(X)))
            points.append(0)
            if verbose:
                print("Cost after 0 iterations: {}".format(costs[-1]))
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            iteration = i + 1
            if (verbose or graph) and (iteration % step == 0 or
                                       iteration == iterations):
                cost = self.cost(Y, self.forward_prop(X))
                costs.append(cost)
                points.append(iteration)
                if verbose:
                    print("Cost after {} iterations: {}".format(
                        iteration, cost))
        if graph:
            plt.plot(points, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
