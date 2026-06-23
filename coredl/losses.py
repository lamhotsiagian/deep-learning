"""
coredl.losses -- loss functions over `Value` graphs.

Runnable companion to Chapter 6 (Loss Functions). All losses return a single
scalar `Value` you can call `.backward()` on.
"""
from __future__ import annotations

from .engine import Value


def mse_loss(preds, targets):
    """Mean squared error (Chapter 6). preds/targets: lists of Value/number."""
    n = len(preds)
    total = sum(((p - t) ** 2 for p, t in zip(preds, targets)), Value(0.0))
    return total * (1.0 / n)


def hinge_loss(preds, targets):
    """SVM-style max-margin loss; targets in {-1, +1} (Chapter 6 sidebar)."""
    n = len(preds)
    total = Value(0.0)
    for p, t in zip(preds, targets):
        total = total + (1 + (-t) * p).relu()  # max(0, 1 - t*p)
    return total * (1.0 / n)
