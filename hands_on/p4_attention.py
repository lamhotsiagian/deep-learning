"""
Project 4 (core mechanics) -- Bahdanau (additive) attention, in pure NumPy.

The full Project 4 in the book is a PyTorch seq2seq translator. The conceptual
heart, which trips people up, is the attention block: how a decoder state scores
every encoder state, turns those scores into a probability distribution
(respecting a padding mask), and reads out a context vector. This script runs
exactly that, with real numbers, so you can see the alignment matrix.

Run:  python p4_attention.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def bahdanau_attention(dec_h, enc_H, mask, Wa, Ua, va):
    """
    dec_h : (B, dh)          decoder hidden state at one step
    enc_H : (B, T, eh)       all encoder hidden states
    mask  : (B, T)           1 for real tokens, 0 for padding
    Returns context (B, eh) and attention weights (B, T).
    Score: e_t = va^T tanh(Wa h_dec + Ua h_enc_t)     (Bahdanau, 2015)
    """
    B, T, eh = enc_H.shape
    dec_proj = (dec_h @ Wa)[:, None, :]                 # (B,1,att)
    enc_proj = enc_H @ Ua                               # (B,T,att)
    scores = np.tanh(dec_proj + enc_proj) @ va          # (B,T) additive score
    scores = np.where(mask == 1, scores, -1e9)          # mask padding -> -inf
    weights = softmax(scores, axis=1)                   # (B,T) sums to 1 per row
    context = np.einsum("bt,bte->be", weights, enc_H)   # weighted sum of enc states
    return context, weights


def main():
    B, T, eh, dh, att = 1, 6, 8, 8, 16
    enc_H = rng.standard_normal((B, T, eh))
    dec_h = rng.standard_normal((B, dh))
    # Last two positions are padding (e.g. a length-4 sentence padded to T=6).
    mask = np.array([[1, 1, 1, 1, 0, 0]])
    Wa = rng.standard_normal((dh, att)) * 0.3
    Ua = rng.standard_normal((eh, att)) * 0.3
    va = rng.standard_normal((att,)) * 0.3

    context, weights = bahdanau_attention(dec_h, enc_H, mask, Wa, Ua, va)
    np.set_printoptions(precision=3, suppress=True)
    print("encoder states shape :", enc_H.shape)
    print("padding mask         :", mask.ravel())
    print("attention weights    :", weights.ravel())
    print("  -> weights sum      =", round(float(weights.sum()), 6))
    print("  -> weight on padded positions (4,5) =", weights.ravel()[4:])
    print("context vector shape :", context.shape)
    print("context vector       :", context.ravel())
    arg = int(weights.argmax())
    print(f"\ndecoder is attending most to encoder position {arg} "
          f"(weight {weights.ravel()[arg]:.3f})")


if __name__ == "__main__":
    main()
