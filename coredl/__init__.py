"""
coredl -- a tiny, dependency-free deep learning framework that accompanies
"Core Deep Learning: A Practitioner's Guide" by Lamhot Siagian.

It implements, in pure Python, the ideas built up across the book:
  - engine.py : scalar reverse-mode autograd        (Chapters 3-4)
  - nn.py     : neurons, layers, MLP, He/Xavier init (Chapters 1, 2, 5, 8)
  - optim.py  : SGD, Momentum, Adam, AdamW           (Chapter 7)
  - losses.py : MSE, hinge                           (Chapter 6)

Run the demos:
    python -m project.train_xor
    python -m project.test_engine
"""
from .engine import Value
from .nn import MLP, Layer, Neuron
from .optim import SGD, Adam
from .losses import mse_loss, hinge_loss

__all__ = [
    "Value", "MLP", "Layer", "Neuron",
    "SGD", "Adam", "mse_loss", "hinge_loss",
]
__version__ = "1.0.0"
