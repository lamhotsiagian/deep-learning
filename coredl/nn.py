"""
coredl.nn -- neurons, layers, and a multilayer perceptron built on `Value`.

Runnable companion to Chapter 1 (Perceptron), Chapter 2 (Multi-Layer Networks),
Chapter 5 (Activations), and Chapter 8 (Weight Initialization).
"""
from __future__ import annotations

import math
import random

from .engine import Value


class Module:
    """Base class: zero gradients and collect parameters (Chapter 12 analogy)."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in: int, activation: str = "relu"):
        # He-style initialization for ReLU, Xavier-style for tanh (Chapter 8).
        if activation == "relu":
            std = math.sqrt(2.0 / n_in)
        else:
            std = math.sqrt(1.0 / n_in)
        self.w = [Value(random.gauss(0.0, std)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        # z = w . x + b   (the perceptron sum, Chapter 1)
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == "relu":
            return z.relu()
        if self.activation == "tanh":
            return z.tanh()
        return z  # linear (e.g. a regression/output unit)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in: int, n_out: int, activation: str = "relu"):
        self.neurons = [Neuron(n_in, activation) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A configurable multilayer perceptron (Chapter 2)."""

    def __init__(self, n_in: int, hidden: list[int], n_out: int,
                 hidden_activation: str = "relu", out_activation: str = "linear"):
        sizes = [n_in] + hidden + [n_out]
        self.layers = []
        for i in range(len(sizes) - 1):
            act = hidden_activation if i < len(sizes) - 2 else out_activation
            self.layers.append(Layer(sizes[i], sizes[i + 1], act))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
