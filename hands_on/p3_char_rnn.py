"""
Project 3 -- A character-level RNN in pure NumPy (min-char-rnn style).

Trains a vanilla recurrent network one character at a time, via
backpropagation-through-time (BPTT, Chapter 16), then generates new text by
sampling from the model with a temperature knob. Pure NumPy, no frameworks.

Run:  python p3_char_rnn.py
"""
from __future__ import annotations

import numpy as np

np.random.seed(1)

# A small self-contained corpus (repeated so the tiny RNN has something to learn).
TEXT = ("deep learning builds intelligence from data. "
        "neurons learn weights. gradients flow backward. ") * 40

chars = sorted(set(TEXT))
vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# Hyperparameters
H = 100          # hidden size
SEQ = 25         # BPTT truncation length
lr = 1e-1

# Parameters (Chapter 16: shared across time steps)
Wxh = np.random.randn(H, vocab) * 0.01
Whh = np.random.randn(H, H) * 0.01
Why = np.random.randn(vocab, H) * 0.01
bh = np.zeros((H, 1))
by = np.zeros((vocab, 1))


def loss_and_grads(inputs, targets, hprev):
    """Forward + BPTT over one sequence. Returns loss, grads, last hidden."""
    xs, hs, ps = {}, {-1: hprev}, {}
    loss = 0.0
    # ---- forward (unroll across time) ----
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab, 1)); xs[t][inputs[t]] = 1          # one-hot
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)        # hidden state
        y = Why @ hs[t] + by
        ps[t] = np.exp(y - y.max()) / np.exp(y - y.max()).sum()    # softmax
        loss += -np.log(ps[t][targets[t], 0] + 1e-12)             # cross-entropy
    # ---- backward through time ----
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = ps[t].copy(); dy[targets[t]] -= 1                     # softmax+CE grad
        dWhy += dy @ hs[t].T; dby += dy
        dh = Why.T @ dy + dhnext
        dhraw = (1 - hs[t] ** 2) * dh                              # tanh'(.)
        dbh += dhraw
        dWxh += dhraw @ xs[t].T
        dWhh += dhraw @ hs[t - 1].T
        dhnext = Whh.T @ dhraw
    for d in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(d, -5, 5, out=d)                                   # gradient clip
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(h, seed_ix, n, temperature=1.0):
    """Generate n characters by feeding each prediction back as the next input."""
    x = np.zeros((vocab, 1)); x[seed_ix] = 1
    out = []
    for _ in range(n):
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        p = np.exp(y / temperature - (y / temperature).max())
        p = (p / p.sum()).ravel()
        ix = np.random.choice(range(vocab), p=p)                  # temperature sampling
        x = np.zeros((vocab, 1)); x[ix] = 1
        out.append(itos[ix])
    return "".join(out)


def evaluate(text):
    """Held-out average per-character loss and next-character accuracy (no grad)."""
    ids = [stoi[c] for c in text]
    h = np.zeros((H, 1)); total_loss, correct, n = 0.0, 0, 0
    for t in range(len(ids) - 1):
        x = np.zeros((vocab, 1)); x[ids[t]] = 1
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        p = np.exp(y - y.max()); p /= p.sum()
        total_loss += -np.log(p[ids[t + 1], 0] + 1e-12)
        correct += int(p.argmax() == ids[t + 1]); n += 1
    return total_loss / n, correct / n


def save_model(path):
    np.savez(path, Wxh=Wxh, Whh=Whh, Why=Why, bh=bh, by=by)


def load_model(path):
    global Wxh, Whh, Why, bh, by
    z = np.load(path)
    Wxh, Whh, Why, bh, by = z["Wxh"], z["Whh"], z["Why"], z["bh"], z["by"]


def main():
    global Wxh, Whh, Why, bh, by
    import os
    import tempfile

    # 1) OBJECTIVE -------------------------------------------------------------
    print("[1 OBJECTIVE] char-level RNN language model; metric=val loss + next-char acc")

    # 2) DATA + 3) PREPROCESS (char tokenization, vocab built above) -----------
    print(f"[2 DATA]      corpus chars={len(TEXT)}")
    print(f"[3 PREPROCESS] vocab={vocab}, BPTT length SEQ={SEQ}, hidden H={H}")

    # 4) SPLIT -- first 85% train text, last 15% validation text ---------------
    cut = int(0.85 * len(TEXT))
    train_text, val_text = TEXT[:cut], TEXT[cut:]
    print(f"[4 SPLIT]     train chars={len(train_text)}  val chars={len(val_text)}")

    # 5) MODEL -----------------------------------------------------------------
    print(f"[5 MODEL]     vanilla RNN ({vocab}->{H}->{vocab}), tanh, Adagrad, grad-clip=5")

    # 6) TRAIN (+val monitoring) and 7) TUNE (keep best-val checkpoint) --------
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by)
    smooth = -np.log(1.0 / vocab) * SEQ
    hprev = np.zeros((H, 1)); p = 0
    best_val, best = np.inf, None
    for n in range(5001):
        if p + SEQ + 1 >= len(train_text):
            hprev = np.zeros((H, 1)); p = 0
        inputs = [stoi[c] for c in train_text[p:p + SEQ]]
        targets = [stoi[c] for c in train_text[p + 1:p + SEQ + 1]]
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = loss_and_grads(inputs, targets, hprev)
        smooth = 0.999 * smooth + 0.001 * loss
        for param, dparam, mem in ((Wxh, dWxh, mWxh), (Whh, dWhh, mWhh),
                                   (Why, dWhy, mWhy), (bh, dbh, mbh), (by, dby, mby)):
            mem += dparam * dparam
            param -= lr * dparam / (np.sqrt(mem) + 1e-8)          # Adagrad
        if n % 1000 == 0 and n > 0:
            vloss, vacc = evaluate(val_text)
            if vloss < best_val:
                best_val = vloss
                best = (Wxh.copy(), Whh.copy(), Why.copy(), bh.copy(), by.copy())
            print(f"[6 TRAIN]     iter {n:4d}  train_loss {smooth/SEQ:.3f}  val_loss {vloss:.3f}  val_acc {vacc:.3f}")
        p += SEQ
    if best:
        Wxh, Whh, Why, bh, by = best
    print(f"[7 TUNE]      restored best-val checkpoint (val_loss {best_val:.3f})")

    # 8) EVALUATE on held-out validation text ----------------------------------
    vloss, vacc = evaluate(val_text)
    print(f"[8 EVALUATE]  val_loss={vloss:.3f}  next_char_acc={vacc:.3f}")

    # 9) DEPLOY (save -> reload) + INFERENCE + MONITOR -------------------------
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "rnn_char.npz")
    save_model(path); load_model(path)                           # serialization round-trip
    gen = sample(np.zeros((H, 1)), stoi["d"], 80, temperature=0.4)
    print(f"[9 DEPLOY]    saved+reloaded; inference (sample, T=0.4):")
    print(f"              {gen!r}")
    tloss, _ = evaluate(train_text[:len(val_text)])
    print(f"[9 MONITOR]   train_loss={tloss:.3f}  val_loss={vloss:.3f}  overfit_gap={vloss - tloss:+.3f}")

    # TEST ---------------------------------------------------------------------
    assert vacc > 0.85, f"val next-char accuracy too low: {vacc:.3f}"
    assert "learning" in gen or "neurons" in gen, f"generation off-pattern: {gen!r}"
    print(f"[TEST]        PASS  (val_acc {vacc:.3f} > 0.85, generation on-pattern)")


if __name__ == "__main__":
    main()
