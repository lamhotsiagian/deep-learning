"""
train_xor.py -- solve XOR with a 2-layer MLP using only `coredl`.

This is the runnable payoff of Chapters 1-7: the perceptron could NOT learn XOR
(Chapter 1), but a multi-layer network trained by backprop (Chapters 2-4) with
Adam (Chapter 7) can. Expect the loss to fall toward zero and all four points to
be classified correctly.

Run:  python -m project.train_xor      (from the repo root)
  or: python train_xor.py              (from the project/ directory)
"""
from __future__ import annotations

import os
import sys

# Allow running directly (python train_xor.py) as well as as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from coredl import MLP, Adam, mse_loss

# The four XOR points and their targets (Chapter 1's canonical non-separable set).
XS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
YS = [0.0, 1.0, 1.0, 0.0]


def main(steps: int = 400, seed: int = 1) -> float:
    random.seed(seed)
    # 2 -> 8 -> 8 -> 1, tanh hidden units, linear output (Chapter 2 / 5).
    model = MLP(2, [8, 8], 1, hidden_activation="tanh", out_activation="linear")
    opt = Adam(model.parameters(), lr=0.05)

    loss_val = float("inf")
    for step in range(steps):
        preds = [model(x) for x in XS]          # forward pass
        loss = mse_loss(preds, YS)              # Chapter 6
        opt.zero_grad()                          # Chapter 4/12: clear grads
        loss.backward()                          # Chapter 3-4: reverse autodiff
        opt.step()                               # Chapter 7: parameter update
        loss_val = loss.data
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss_val:.6f}")

    print("\nfinal predictions vs targets:")
    correct = 0
    for x, y in zip(XS, YS):
        pred = model(x).data
        label = 1 if pred >= 0.5 else 0
        correct += int(label == y)
        print(f"  {x} -> {pred:+.3f}  (label {label}, target {int(y)})")
    print(f"\naccuracy: {correct}/4")
    return loss_val


if __name__ == "__main__":
    main()
