"""
Project 4 -- A trainable sequence-to-sequence model with attention, in NumPy.

A full end-to-end pipeline. The task is sequence reversal, which is solvable
*only* by attending to the right source position at each output step, so the
learned attention matrix becomes anti-diagonal (Chapter 17). Bahdanau (additive)
attention and every gradient are implemented by hand and gradient-checked:
  1 OBJECTIVE -> 2 DATA -> 3 PREPROCESS -> 4 SPLIT -> 5 MODEL ->
  6 TRAIN(+val) -> 7 TUNE -> 8 EVALUATE -> 9 DEPLOY+MONITOR -> TEST

(The full PyTorch RNN-encoder/decoder version is the chapter code; the Bahdanau
forward demo is in p4_attention.py.)

Run:  python p4_seq2seq.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)
VOC = 8          # symbol vocabulary
L = 5            # sequence length
D = 16           # embedding / state dim
DA = 16          # attention dim


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


def init_params():
    s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(1.0 / a)
    return {
        "We": s(VOC, D),          # source-token embedding (encoder states = emb + pos)
        "Pe": s(L, D),            # encoder positional embedding
        "Qe": s(L, D),            # decoder per-output-position query embedding
        "Wq": s(D, DA), "Wk": s(D, DA), "v": rng.standard_normal(DA) * 0.1,
        "Wo": s(D, VOC), "bo": np.zeros(VOC),
    }


def forward(P, src):
    """src: (L,) ids. Returns logits (L,VOC), attention A (L,L), and a cache."""
    H = P["We"][src] + P["Pe"]                      # encoder states (L, D)
    Q = P["Qe"]                                     # decoder queries  (L, D)
    proj = Q @ P["Wq"]                              # (L, DA)
    projH = H @ P["Wk"]                             # (L, DA)
    u = proj[:, None, :] + projH[None, :, :]        # (Lout, Lenc, DA)
    g = np.tanh(u)
    scores = g @ P["v"]                             # (Lout, Lenc)
    A = softmax(scores, axis=1)
    context = A @ H                                 # (Lout, D)
    logits = context @ P["Wo"] + P["bo"]            # (Lout, VOC)
    return logits, A, (src, H, Q, proj, projH, g, A, context)


def loss_and_grads(P, src, tgt):
    logits, A, cache = forward(P, src)
    src, H, Q, proj, projH, g, A, context = cache
    probs = softmax(logits, axis=1)
    loss = -np.log(probs[np.arange(L), tgt] + 1e-12).mean()
    g_ = {k: np.zeros_like(v) for k, v in P.items()}
    dlogits = probs.copy(); dlogits[np.arange(L), tgt] -= 1; dlogits /= L
    g_["Wo"] = context.T @ dlogits; g_["bo"] = dlogits.sum(0)
    dcontext = dlogits @ P["Wo"].T                  # (Lout, D)
    dA = dcontext @ H.T                             # (Lout, Lenc)
    dH = A.T @ dcontext                             # (Lenc, D)  via context
    dscores = A * (dA - (dA * A).sum(1, keepdims=True))   # softmax backward
    g_["v"] = np.einsum("tj,tjk->k", dscores, g)
    dg = dscores[:, :, None] * P["v"][None, None, :]
    du = dg * (1 - g ** 2)                          # tanh'
    dproj = du.sum(axis=1)                          # (Lout, DA)
    dprojH = du.sum(axis=0)                         # (Lenc, DA)
    g_["Wq"] = Q.T @ dproj
    g_["Qe"] = dproj @ P["Wq"].T                    # queries are learned params
    g_["Wk"] = H.T @ dprojH
    dH = dH + dprojH @ P["Wk"].T                    # (Lenc, D) via scores
    g_["Pe"] += dH
    np.add.at(g_["We"], src, dH)
    return loss, g_, A


def gradient_check(P, src, tgt, key, eps=1e-5):
    _, g_, _ = loss_and_grads(P, src, tgt)
    arr = P[key]; flat = arr.ravel(); gflat = g_[key].ravel(); rels = []
    for _ in range(8):
        i = rng.integers(flat.size); o = flat[i]
        flat[i] = o + eps; lp = -np.log(softmax(forward(P, src)[0], 1)[np.arange(L), tgt] + 1e-12).mean()
        flat[i] = o - eps; lm = -np.log(softmax(forward(P, src)[0], 1)[np.arange(L), tgt] + 1e-12).mean()
        flat[i] = o
        num = (lp - lm) / (2 * eps)
        rels.append(abs(num - gflat[i]) / (abs(num) + abs(gflat[i]) + 1e-12))
    return max(rels)


def batch(n):
    src = rng.integers(0, VOC, (n, L))
    tgt = src[:, ::-1].copy()                       # target = reversed source
    return src, tgt


def accuracy(P, src, tgt):
    seq_ok = 0
    for s, t in zip(src, tgt):
        pred = forward(P, s)[0].argmax(1)
        seq_ok += int((pred == t).all())
    return seq_ok / len(src)


def main():
    print("[1 OBJECTIVE] seq2seq reversal via attention; metric=full-sequence accuracy")

    # 2) DATA + 3) PREPROCESS
    Xtr, Ytr = batch(2000); Xva, Yva = batch(400); Xte, Yte = batch(400)
    print(f"[2 DATA]      random sequences, vocab={VOC}, length L={L}")
    print(f"[3 PREPROCESS] integer tokens; target = reverse(source)")

    # 4) SPLIT
    print(f"[4 SPLIT]     train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # 5) MODEL + gradient check
    P = init_params()
    s0, t0 = Xtr[0], Ytr[0]
    rel = max(gradient_check(P, s0, t0, k) for k in ["Wq", "Wk", "v", "Wo", "Qe", "We", "Pe"])
    print("[5 MODEL]     emb encoder + Bahdanau attention + linear decoder")
    print(f"[5 MODEL]     attention gradient check max rel err {rel:.2e} -> {'PASS' if rel < 1e-4 else 'FAIL'}")

    # 6) TRAIN + 7) TUNE
    st = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in P.items()}
    lr, b1, b2, t = 5e-3, 0.9, 0.999, 0
    best_va, best = 0.0, None
    for epoch in range(1, 31):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 32):
            grads = {k: np.zeros_like(v) for k, v in P.items()}; bl = 0.0
            bi = order[s:s + 32]
            for j in bi:
                loss, g_, _ = loss_and_grads(P, Xtr[j], Ytr[j])
                for k in P: grads[k] += g_[k]
                bl += loss
            t += 1
            for k in P:
                grads[k] /= len(bi)
                st[k]["m"] = b1 * st[k]["m"] + (1 - b1) * grads[k]
                st[k]["v"] = b2 * st[k]["v"] + (1 - b2) * grads[k] ** 2
                mh = st[k]["m"] / (1 - b1 ** t); vh = st[k]["v"] / (1 - b2 ** t)
                P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        va = accuracy(P, Xva, Yva)
        if va > best_va: best_va, best = va, {k: v.copy() for k, v in P.items()}
        if epoch % 5 == 0:
            print(f"[6 TRAIN]     epoch {epoch:2d}  val_seq_acc {va:.3f}")
        if best_va >= 0.999:
            break
    P = best
    print(f"[7 TUNE]      restored best-val checkpoint (val_seq_acc {best_va:.3f})")

    # 8) EVALUATE
    test_acc = accuracy(P, Xte, Yte)
    print(f"[8 EVALUATE]  test_seq_acc={test_acc:.3f}")

    # 9) DEPLOY + INFERENCE + MONITOR (check the attention is anti-diagonal)
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "seq2seq.npz")
    np.savez(path, **P); z = np.load(path); served = {k: z[k] for k in P}
    demo = np.array([1, 2, 3, 4, 5]) % VOC
    logits, A, _ = forward(served, demo)
    pred = logits.argmax(1)
    print(f"[9 DEPLOY]    saved+reloaded; reverse {demo.tolist()} -> {pred.tolist()} (true {demo[::-1].tolist()})")
    anti_diag = float(np.mean(A.argmax(1) == np.arange(L)[::-1]))   # attends to reversed pos?
    print(f"[9 MONITOR]   attention anti-diagonal alignment={anti_diag:.2f}  (1.0 = perfect reversal attention)")

    # TEST
    assert rel < 1e-4, f"gradient check failed: {rel:.2e}"
    assert test_acc > 0.95, f"test sequence accuracy too low: {test_acc:.3f}"
    assert anti_diag == 1.0, f"attention not anti-diagonal: {anti_diag:.2f}"
    print(f"[TEST]        PASS  (grad-check {rel:.1e}, test_acc {test_acc:.3f} > 0.95, attention anti-diagonal)")


if __name__ == "__main__":
    main()
