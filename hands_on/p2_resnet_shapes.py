"""
Project 2 (core mechanics) -- ResNet spatial arithmetic and the residual block,
verified in pure NumPy.

The full Project 2 in the book is a PyTorch ResNet trained on CIFAR-10. PyTorch
is not always installable, so this script proves the two pieces that confuse
people most, and that the PyTorch code mirrors exactly:

  (1) the conv output-size formula  W_out = (W_in - F + 2P) / S + 1
  (2) the residual block, including the 1x1 stride-2 *projection shortcut* that
      lets identity be added when the main path downsamples and changes channels.

Run:  python p2_resnet_shapes.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def conv2d(x, w, stride=1, pad=1):
    """Naive NCHW convolution. x:(N,Cin,H,W)  w:(Cout,Cin,F,F) -> (N,Cout,Ho,Wo)."""
    N, Cin, H, W = x.shape
    Cout, _, F, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H - F + 2 * pad) // stride + 1                # the output-size formula
    Wo = (W - F + 2 * pad) // stride + 1
    out = np.zeros((N, Cout, Ho, Wo))
    for i in range(Ho):
        for j in range(Wo):
            r, c = i * stride, j * stride
            patch = xp[:, :, r:r + F, c:c + F]          # (N,Cin,F,F)
            out[:, :, i, j] = np.tensordot(patch, w, axes=([1, 2, 3], [1, 2, 3]))
    return out


def relu(x):
    return np.maximum(0, x)


def basic_block(x, c_out, stride):
    """A ResNet BasicBlock forward. Downsamples when stride=2 and projects the
    identity with a 1x1 conv so shapes match for the residual add."""
    c_in = x.shape[1]
    w1 = rng.standard_normal((c_out, c_in, 3, 3)) * 0.05    # 3x3, may downsample
    w2 = rng.standard_normal((c_out, c_out, 3, 3)) * 0.05   # 3x3, stride 1
    out = relu(conv2d(x, w1, stride=stride, pad=1))
    out = conv2d(out, w2, stride=1, pad=1)

    # --- the shortcut ---
    if stride != 1 or c_in != c_out:
        w_sc = rng.standard_normal((c_out, c_in, 1, 1)) * 0.05
        identity = conv2d(x, w_sc, stride=stride, pad=0)    # 1x1 stride-2 projection
        shortcut = "1x1 conv projection"
    else:
        identity = x                                        # plain identity
        shortcut = "identity"
    out = relu(out + identity)                              # the residual add
    return out, shortcut


def main():
    # Output-size formula sanity checks (Chapter 15).
    out_size = lambda I, K, P, S: (I - K + 2 * P) // S + 1
    print("output-size formula checks:")
    for I, K, P, S in [(32, 3, 1, 1), (32, 3, 1, 2), (32, 7, 3, 2)]:
        print(f"  I={I} K={K} P={P} S={S}  ->  O={out_size(I, K, P, S)}")

    # A ResNet-18-style stage progression on a CIFAR-sized tensor.
    x = rng.standard_normal((2, 3, 32, 32))                 # batch 2, RGB, 32x32
    print(f"\ninput               {x.shape}")
    # stem: 3x3 conv keeps 32x32, lifts to 64 channels
    stem_w = rng.standard_normal((64, 3, 3, 3)) * 0.05
    x = relu(conv2d(x, stem_w, stride=1, pad=1))
    print(f"after stem (64ch)   {x.shape}")
    for c_out, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
        x, sc = basic_block(x, c_out, stride)
        print(f"block c={c_out:3d} s={stride}   {x.shape}   shortcut: {sc}")
    pooled = x.mean(axis=(2, 3))                            # global average pool
    print(f"after global pool   {pooled.shape}   -> feeds a Linear({pooled.shape[1]}, 10)")


if __name__ == "__main__":
    main()
