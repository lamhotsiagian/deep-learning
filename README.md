# coredl — the book's runnable mini-framework

`coredl` is a tiny, **dependency-free** deep learning framework that accompanies
*Core Deep Learning: A Practitioner's Guide*. It implements, in pure Python, the
exact ideas the book builds up chapter by chapter, so you can read a chapter and
then run the corresponding code.

## Layout

| File | What it implements | Chapters |
|------|--------------------|----------|
| `coredl/engine.py` | Scalar reverse-mode autograd (`Value`) | 3, 4 |
| `coredl/nn.py` | `Neuron`, `Layer`, `MLP`, He/Xavier init | 1, 2, 5, 8 |
| `coredl/optim.py` | `SGD` (+ momentum), `Adam`/`AdamW` | 7 |
| `coredl/losses.py` | `mse_loss`, `hinge_loss` | 6 |
| `train_xor.py` | Trains an MLP to solve XOR end-to-end | 1–7 |
| `test_engine.py` | Gradient checks (numerical vs engine) | 3, 4 |

## Run it

No installation, no dependencies — just Python 3.8+.

```bash
# from the repository root
python -m project.test_engine     # gradient checks (should print ALL TESTS PASSED)
python -m project.train_xor       # trains XOR, should reach accuracy 4/4

# or from inside the project/ directory
python test_engine.py
python train_xor.py
```

## Expected output

`test_engine.py` ends with `ALL TESTS PASSED`. `train_xor.py` drives the loss to
~0 and prints `accuracy: 4/4` — the concrete proof that a multi-layer network
(Chapter 2) trained by backpropagation (Chapters 3–4) and Adam (Chapter 7) solves
the very problem a single perceptron could not (Chapter 1).

## Why scalar-valued?

For learning, not speed. Operating one number at a time makes every gradient
inspectable and removes all "magic" from autograd. Chapters 12–13 show how
PyTorch and JAX scale the *same* algorithm to GPU tensors — the math is identical;
only the engineering differs.
# deep-learning
