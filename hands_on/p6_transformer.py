"""
Project 6 -- A GPT-style Transformer language model in pure NumPy.

A complete, trainable, single-layer causal Transformer with manual
backpropagation: token + positional embeddings, masked (causal) self-attention,
a residual + feed-forward block, and an unembedding to vocabulary logits. We
gradient-check the hand-derived backward pass against numerical gradients
(Chapter 3), then train on a tiny corpus and generate text.

This is the from-scratch payoff of Chapter 17: the same softmax(QK^T/sqrt(d))V
you read about, now trained end to end by gradients you can inspect.

Run:  python p6_transformer.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)

# --- tiny corpus (highly repetitive so one layer can learn it fast) ---------
TEXT = "deep learning transformers attend. " * 60
chars = sorted(set(TEXT))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in TEXT])

# --- hyperparameters --------------------------------------------------------
T = 16          # context length (block size)
D = 32          # model dimension
DFF = 64        # feed-forward hidden dimension
SCALE = 1.0 / np.sqrt(D)

# causal mask: True where attention is forbidden (upper triangle)
CAUSAL = np.triu(np.ones((T, T), dtype=bool), k=1)


def init(*shape, s=0.02):
    return rng.standard_normal(shape) * s


# --- parameters -------------------------------------------------------------
P = {
    "We": init(V, D),  "Wp": init(T, D),                       # embeddings
    "Wq": init(D, D),  "Wk": init(D, D), "Wv": init(D, D), "Wo": init(D, D),
    "W1": init(D, DFF), "b1": np.zeros(DFF),
    "W2": init(DFF, D), "b2": np.zeros(D),
    "Wu": init(D, V),  "bu": np.zeros(V),                      # unembedding
}


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(idx, targets=None):
    """idx, targets: (T,) int arrays. Returns loss and a cache for backward."""
    emb = P["We"][idx] + P["Wp"]                      # (T, D)
    Q, K, Vv = emb @ P["Wq"], emb @ P["Wk"], emb @ P["Wv"]
    scores = (Q @ K.T) * SCALE                        # (T, T)
    scores = np.where(CAUSAL, -1e9, scores)           # causal mask
    A = softmax(scores, axis=1)                       # (T, T) attention weights
    attn = A @ Vv                                     # (T, D)
    x1 = emb + attn @ P["Wo"]                          # residual 1
    h = np.maximum(0, x1 @ P["W1"] + P["b1"])         # FFN hidden (ReLU)
    x2 = x1 + h @ P["W2"] + P["b2"]                    # residual 2
    logits = x2 @ P["Wu"] + P["bu"]                    # (T, V)
    cache = (idx, emb, Q, K, Vv, A, attn, x1, h, x2, logits)
    if targets is None:
        return logits, cache
    probs = softmax(logits, axis=1)
    loss = -np.log(probs[np.arange(T), targets] + 1e-12).mean()
    return loss, cache


def backward(cache, targets):
    idx, emb, Q, K, Vv, A, attn, x1, h, x2, logits = cache
    g = {k: np.zeros_like(v) for k, v in P.items()}
    # cross-entropy + softmax
    dlogits = softmax(logits, axis=1)
    dlogits[np.arange(T), targets] -= 1
    dlogits /= T
    g["Wu"] = x2.T @ dlogits; g["bu"] = dlogits.sum(0)
    dx2 = dlogits @ P["Wu"].T
    # FFN block (residual: dx2 flows to x1 and through the FFN)
    dx1 = dx2.copy()
    g["W2"] = h.T @ dx2; g["b2"] = dx2.sum(0)
    dh = (dx2 @ P["W2"].T) * (h > 0)                  # ReLU'
    g["W1"] = x1.T @ dh; g["b1"] = dh.sum(0)
    dx1 += dh @ P["W1"].T
    # attention block (residual: dx1 flows to emb and through attention)
    demb = dx1.copy()
    dao = dx1
    g["Wo"] = attn.T @ dao
    dattn = dao @ P["Wo"].T
    dA = dattn @ Vv.T
    dV = A.T @ dattn
    # softmax backward (row-wise); masked entries have A=0 so contribute 0
    dscores = A * (dA - (dA * A).sum(axis=1, keepdims=True))
    dscores *= SCALE
    dQ = dscores @ K
    dK = dscores.T @ Q
    g["Wq"] = emb.T @ dQ; g["Wk"] = emb.T @ dK; g["Wv"] = emb.T @ dV
    demb += dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
    # embeddings
    np.add.at(g["We"], idx, demb)
    g["Wp"] += demb
    return g


def gradient_check(name="Wq", eps=1e-5):
    """Verify a parameter's hand-derived gradient against central differences.

    Gradient checking validates the *implementation*, so we evaluate at a
    healthy-scale random weight snapshot (at the tiny 0.02 training init the
    attention scores are ~0, the softmax is flat, and the true Wq/Wk gradients
    are ~1e-8, too small to resolve numerically). We restore the weights after.
    """
    saved = {k: v.copy() for k, v in P.items()}
    for k in P:                                   # moderate scale: scores ~O(0.1)
        P[k][...] = rng.standard_normal(P[k].shape) * 0.2  # keeps softmax responsive

    i0 = 100
    idx, tgt = data[i0:i0 + T], data[i0 + 1:i0 + T + 1]
    _, cache = forward(idx, tgt)
    g = backward(cache, tgt)
    rels = []
    for _ in range(8):
        a, b = rng.integers(P[name].shape[0]), rng.integers(P[name].shape[1])
        orig = P[name][a, b]
        P[name][a, b] = orig + eps; lp, _ = forward(idx, tgt)
        P[name][a, b] = orig - eps; lm, _ = forward(idx, tgt)
        P[name][a, b] = orig
        num = (lp - lm) / (2 * eps)
        rels.append(abs(num - g[name][a, b]) / (abs(num) + abs(g[name][a, b]) + 1e-12))
    for k in P:                                   # restore the training init
        P[k][...] = saved[k]
    return max(rels)


# --- Adam optimizer ---------------------------------------------------------
M = {k: np.zeros_like(v) for k, v in P.items()}
Vd = {k: np.zeros_like(v) for k, v in P.items()}


def adam_step(grads, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in P:
        M[k] = b1 * M[k] + (1 - b1) * grads[k]
        Vd[k] = b2 * Vd[k] + (1 - b2) * grads[k] ** 2
        mh = M[k] / (1 - b1 ** t); vh = Vd[k] / (1 - b2 ** t)
        P[k] -= lr * mh / (np.sqrt(vh) + eps)


def generate(prompt, n):
    ctx = [stoi[c] for c in prompt]
    out = list(prompt)
    for _ in range(n):
        window = ctx[-T:]
        L = len(window)
        padded = window + [0] * (T - L)                  # right-pad to T
        logits, _ = forward(np.array(padded))
        # Causal mask makes the right-pad "future" tokens invisible to position
        # L-1, and the real tokens keep their correct positional embeddings.
        nxt = int(softmax(logits[L - 1]).argmax())       # greedy next token
        out.append(itos[nxt]); ctx.append(nxt)
    return "".join(out)


def evaluate(positions):
    """Average next-token loss and next-token accuracy over given window starts."""
    losses, correct, total = [], 0, 0
    for i0 in positions:
        idx, tgt = data[i0:i0 + T], data[i0 + 1:i0 + T + 1]
        loss, cache = forward(idx, tgt)
        losses.append(loss)
        logits = cache[-1]
        correct += int((logits.argmax(1) == tgt).sum()); total += T
    return float(np.mean(losses)), correct / total


def save_model(path):
    np.savez(path, **P)


def load_model(path):
    z = np.load(path)
    for k in P:
        P[k][...] = z[k]


def main():
    import os
    import tempfile

    # 1) OBJECTIVE -------------------------------------------------------------
    print("[1 OBJECTIVE] char-level language model; metric=val loss + next-char acc")

    # 2) DATA + 3) PREPROCESS (tokenize to char ids; vocab built above) --------
    print(f"[2 DATA]      corpus chars={len(TEXT)}")
    print(f"[3 PREPROCESS] vocab={V}, context T={T}, d_model={D} (char tokenization)")

    # 4) SPLIT -- train on the first 85% of positions, validate on the last 15%
    n_pos = len(data) - T - 1
    cut = int(0.85 * n_pos)
    train_pos = np.arange(0, cut)
    val_pos = np.arange(cut, n_pos)
    print(f"[4 SPLIT]     train windows={len(train_pos)}  val windows={len(val_pos)}")

    # 5) MODEL -----------------------------------------------------------------
    print("[5 MODEL]     1-layer causal Transformer (token+pos emb, masked attn, FFN, unembed)")
    rel = gradient_check()
    status = "PASS" if rel < 1e-4 else "FAIL"
    print(f"[5 MODEL]     gradient check (Wq) max rel err {rel:.2e} -> {status}")

    # 6) TRAIN (+val monitoring) and 7) TUNE (LR step decay, keep best val) ----
    STEPS = 5000
    best_val, best_snap = np.inf, None
    for step in range(1, STEPS + 1):
        i0 = int(rng.choice(train_pos))
        idx, tgt = data[i0:i0 + T], data[i0 + 1:i0 + T + 1]
        loss, cache = forward(idx, tgt)
        lr = 3e-3 if step < STEPS // 2 else 1e-3              # step decay
        adam_step(backward(cache, tgt), step, lr=lr)
        if step % 1000 == 0:
            vloss, vacc = evaluate(val_pos[::7])
            if vloss < best_val:
                best_val, best_snap = vloss, {k: v.copy() for k, v in P.items()}
            print(f"[6 TRAIN]     step {step:4d}  train_loss {loss:.4f}  val_loss {vloss:.4f}  val_acc {vacc:.3f}")
    if best_snap:                                            # restore best (early-stop style)
        for k in P:
            P[k][...] = best_snap[k]
    print(f"[7 TUNE]      restored best checkpoint (val_loss {best_val:.4f})")

    # 8) EVALUATE on held-out val ----------------------------------------------
    vloss, vacc = evaluate(val_pos)
    print(f"[8 EVALUATE]  val_loss={vloss:.4f}  next_char_acc={vacc:.3f}")

    # 9) DEPLOY (save -> reload) + MONITOR + INFERENCE -------------------------
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "gpt_char.npz")
    save_model(path); load_model(path)                       # serialization round-trip
    sample = generate("deep ", 60)
    print(f"[9 DEPLOY]    saved+reloaded; inference (greedy, prompt='deep '):")
    print(f"              {sample!r}")
    train_loss, _ = evaluate(train_pos[::13])
    gap = vloss - train_loss
    print(f"[9 MONITOR]   train_loss={train_loss:.4f}  val_loss={vloss:.4f}  overfit_gap={gap:+.4f}")

    # TEST ---------------------------------------------------------------------
    assert rel < 1e-4, f"gradient check failed: {rel:.2e}"
    assert vacc > 0.80, f"val next-char accuracy too low: {vacc:.3f}"
    assert "learning transformers" in sample, f"generation off-pattern: {sample!r}"
    print(f"[TEST]        PASS  (grad-check {rel:.1e}, val_acc {vacc:.3f} > 0.80, generation on-pattern)")


if __name__ == "__main__":
    main()
