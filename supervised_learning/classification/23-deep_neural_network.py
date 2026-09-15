#!/usr/bin/env python3
"""Adds progress reporting and plotting to a deep network."""

import matplotlib.pyplot as plt
from importlib import import_module

_Base = import_module('22-deep_neural_network').DeepNeuralNetwork


class DeepNeuralNetwork(_Base):
    """A trainable and observable deep sigmoid network."""

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Train, report cost, optionally plot, and evaluate."""
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
        costs, points = [], []
        if verbose or graph:
            A, _ = self.forward_prop(X)
            costs.append(self.cost(Y, A))
            points.append(0)
            if verbose:
                print("Cost after 0 iterations: {}".format(costs[-1]))
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            iteration = i + 1
            if (verbose or graph) and (iteration % step == 0 or
                                       iteration == iterations):
                A, _ = self.forward_prop(X)
                cost = self.cost(Y, A)
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
