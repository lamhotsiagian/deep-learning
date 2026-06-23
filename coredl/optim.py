"""
coredl.optim -- optimizers operating on a list of `Value` parameters.

Runnable companion to Chapter 7 (Optimizers: SGD, Momentum, Adam, AdamW).
"""
from __future__ import annotations

import math


class SGD:
    """Stochastic gradient descent with optional momentum (Chapter 7)."""

    def __init__(self, params, lr=0.05, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self._v = [0.0 for _ in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            self._v[i] = self.momentum * self._v[i] + p.grad
            p.data -= self.lr * self._v[i]

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0


class Adam:
    """Adam / AdamW with bias correction (Chapter 7).

    Set `weight_decay > 0` for decoupled (AdamW-style) weight decay.
    """

    def __init__(self, params, lr=1e-2, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self._m = [0.0 for _ in self.params]
        self._v = [0.0 for _ in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self._m[i] = self.b1 * self._m[i] + (1 - self.b1) * g
            self._v[i] = self.b2 * self._v[i] + (1 - self.b2) * g * g
            m_hat = self._m[i] / (1 - self.b1 ** self.t)
            v_hat = self._v[i] / (1 - self.b2 ** self.t)
            # Decoupled weight decay (AdamW): applied directly to the parameter.
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0
