"""
coredl.engine -- a minimal scalar-valued automatic differentiation engine.

This is the runnable companion to Chapter 3 (Backpropagation from Scratch) and
Chapter 4 (Build Your Own Mini Framework). Every `Value` wraps a single number,
remembers how it was produced, and knows how to push gradient to its parents.
Pure Python, no dependencies, so it runs anywhere.
"""
from __future__ import annotations

import math


class Value:
    """A node in a dynamically-built computational graph (a DAG)."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        # Internal autograd bookkeeping (Chapter 4).
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- forward operations, each attaching its local backward closure --------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> accumulate (Chapter 4)
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int/float powers supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad  # n*x^(n-1)

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad  # 1 - tanh^2

        out._backward = _backward
        return out

    # -- the backward pass: reverse topological order (Chapter 4) -------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0  # seed dL/dL = 1
        for v in reversed(topo):
            v._backward()

    # -- convenience operators so Value behaves like a number -----------------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
