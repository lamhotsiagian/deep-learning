"""
test_engine.py -- gradient checks for the coredl autograd engine.

Runnable companion to Chapter 3 (gradient checking) and Chapter 4 (verify your
engine). Compares engine gradients against central-difference numerical
gradients. Exits non-zero if any check fails, so it doubles as a CI smoke test.

Run:  python -m project.test_engine
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coredl import Value


def numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)  # central difference


def check(name, ok):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    return ok


def main() -> int:
    all_ok = True

    # 1) c = a*b + a**2  ->  dc/da = b + 2a, dc/db = a
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a ** 2
    c.backward()
    da = numerical_grad(lambda av: av * (-3.0) + av ** 2, 2.0)
    db = numerical_grad(lambda bv: 2.0 * bv + 4.0, -3.0)
    all_ok &= check("mul+pow dc/da", abs(a.grad - da) < 1e-4)
    all_ok &= check("mul+pow dc/db", abs(b.grad - db) < 1e-4)

    # 2) tanh
    x = Value(0.7)
    y = x.tanh()
    y.backward()
    import math
    expected = 1 - math.tanh(0.7) ** 2
    all_ok &= check("tanh derivative", abs(x.grad - expected) < 1e-4)

    # 3) relu on both sides of zero
    xp = Value(1.5); xp.relu().backward()
    xn = Value(-1.5); xn.relu().backward()
    all_ok &= check("relu grad (x>0)", abs(xp.grad - 1.0) < 1e-9)
    all_ok &= check("relu grad (x<0)", abs(xn.grad - 0.0) < 1e-9)

    # 4) gradient accumulation on a shared node (the +=, not = lesson)
    s = Value(3.0)
    out = s * s + s  # ds = 2s + 1 = 7
    out.backward()
    all_ok &= check("shared-node accumulation", abs(s.grad - 7.0) < 1e-9)

    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
