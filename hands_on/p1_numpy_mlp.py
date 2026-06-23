"""
Project 1 -- A Multi-Layer Perceptron in pure NumPy.

A COMPLETE end-to-end deep-learning pipeline, no autograd, every gradient by hand:
  1 OBJECTIVE -> 2 DATA -> 3 PREPROCESS -> 4 SPLIT -> 5 MODEL ->
  6 TRAIN(+val) -> 7 TUNE(early stop) -> 8 EVALUATE -> 9 DEPLOY+MONITOR -> TEST
Exits non-zero if the final assertions fail.

Run:  python p1_numpy_mlp.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Model: explicit forward + backward
# ---------------------------------------------------------------------------
def he(n_in, n_out):
    return rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(p, y):
    return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()


class MLP:
    def __init__(self, sizes):
        self.W = [he(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.b = [np.zeros((1, sizes[i + 1])) for i in range(len(sizes) - 1)]

    def forward(self, X):
        self.cache, a = [X], X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]; a = relu(z); self.cache += [z, a]
        logits = a @ self.W[-1] + self.b[-1]; self.cache.append(logits)
        return logits

    def backward(self, X, y, probs, opt):
        n = X.shape[0]
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        dz = probs.copy(); dz[np.arange(n), y] -= 1; dz /= n
        gW[-1] = self.cache[-2].T @ dz; gb[-1] = dz.sum(0, keepdims=True)
        da = dz @ self.W[-1].T
        for i in range(len(self.W) - 2, -1, -1):
            dz = da * (self.cache[1 + 2 * i] > 0)
            gW[i] = self.cache[2 * i].T @ dz; gb[i] = dz.sum(0, keepdims=True)
            da = dz @ self.W[i].T
        opt.step(self.W, self.b, gW, gb)

    def proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return self.forward(X).argmax(1)

    def snapshot(self):
        return ([w.copy() for w in self.W], [b.copy() for b in self.b])

    def restore(self, snap):
        self.W = [w.copy() for w in snap[0]]; self.b = [b.copy() for b in snap[1]]


class Adam:
    def __init__(self, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.s = None

    def step(self, W, b, gW, gb):
        if self.s is None:
            self.s = [[np.zeros_like(p) for p in W + b] for _ in range(2)]
        self.t += 1
        params, grads = W + b, gW + gb
        for i, (p, g) in enumerate(zip(params, grads)):
            self.s[0][i] = self.b1 * self.s[0][i] + (1 - self.b1) * g
            self.s[1][i] = self.b2 * self.s[1][i] + (1 - self.b2) * g ** 2
            mh = self.s[0][i] / (1 - self.b1 ** self.t)
            vh = self.s[1][i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def make_spiral(per_class=120, n_classes=3):
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.0, 1.0, per_class)
        t = np.linspace(c * 4, (c + 1) * 4, per_class) + rng.normal(0, 0.2, per_class)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y.append(np.full(per_class, c))
    return np.vstack(X), np.concatenate(y)


def save_model(path, net, mu, sd):
    d = {f"W{i}": w for i, w in enumerate(net.W)}
    d.update({f"b{i}": b for i, b in enumerate(net.b)})
    d["mu"], d["sd"] = mu, sd
    np.savez(path, **d)


def load_model(path):
    z = np.load(path)
    net = MLP([2, 64, 32, 3])
    net.W = [z[f"W{i}"] for i in range(3)]; net.b = [z[f"b{i}"] for i in range(3)]
    return net, z["mu"], z["sd"]


def main():
    # 1) OBJECTIVE -------------------------------------------------------------
    print("[1 OBJECTIVE] classify the 3-class spiral; metric=test accuracy; target>=0.90")

    # 2) DATA ACQUISITION ------------------------------------------------------
    X, y = make_spiral(per_class=120)
    print(f"[2 DATA]      ingested {len(y)} samples, 2 features, 3 classes")

    # 4) SPLIT (do this before preprocessing to avoid leakage) -----------------
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    n1, n2 = int(0.70 * len(y)), int(0.85 * len(y))
    Xtr, ytr = X[:n1], y[:n1]; Xva, yva = X[n1:n2], y[n1:n2]; Xte, yte = X[n2:], y[n2:]
    print(f"[4 SPLIT]     train={len(ytr)}  val={len(yva)}  test={len(yte)}")

    # 3) PREPROCESS: standardize with TRAIN stats only (augment with jitter) ---
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    std = lambda A: (A - mu) / sd
    Xtr_s, Xva_s, Xte_s = std(Xtr), std(Xva), std(Xte)
    print(f"[3 PREPROCESS] standardized with train stats mu={mu.round(2)} sd={sd.round(2)}")

    # 5) MODEL -----------------------------------------------------------------
    net, opt = MLP([2, 64, 32, 3]), Adam(lr=2e-2)
    print("[5 MODEL]     MLP 2-64-32-3, ReLU hidden, softmax head, Adam")

    # 6) TRAIN + 7) TUNE (early stopping on val accuracy) ----------------------
    best_va, best_snap, patience, wait = 0.0, None, 8, 0
    for epoch in range(1, 1001):
        # data augmentation: small Gaussian jitter each epoch (regularizer)
        Xaug = Xtr_s + rng.normal(0, 0.02, Xtr_s.shape)
        net.backward(Xaug, ytr, softmax(net.forward(Xaug)), opt)
        if epoch % 20 == 0:
            va = (net.predict(Xva_s) == yva).mean()
            if va > best_va:
                best_va, best_snap, wait = va, net.snapshot(), 0
            else:
                wait += 1
            if epoch % 200 == 0:
                tr_loss = cross_entropy(softmax(net.forward(Xtr_s)), ytr)
                print(f"[6 TRAIN]     epoch {epoch:4d}  train_loss {tr_loss:.4f}  val_acc {va:.3f}")
            if wait >= patience:
                print(f"[7 TUNE]      early stop at epoch {epoch} (best val_acc {best_va:.3f})")
                break
    net.restore(best_snap)

    # 8) EVALUATE on the untouched test set ------------------------------------
    test_acc = (net.predict(Xte_s) == yte).mean()
    print(f"[8 EVALUATE]  test_acc={test_acc:.3f}")

    # 9) DEPLOY (serialize -> reload) + MONITOR --------------------------------
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "mlp_spiral.npz")
    save_model(path, net, mu, sd)
    served, smu, ssd = load_model(path)                     # fresh process would do this
    new_points = np.array([[0.0, 0.8], [-0.7, -0.2], [0.5, -0.4]])
    proba = served.proba((new_points - smu) / ssd)
    preds, conf = proba.argmax(1), proba.max(1)
    print(f"[9 DEPLOY]    saved+reloaded model; inference on {len(new_points)} new points -> {preds.tolist()}")
    # MONITOR: input drift (feature-mean shift) + low-confidence rate
    drift = float(np.abs((Xte.mean(0) - mu) / sd).max())
    low_conf = float((conf < 0.6).mean())
    print(f"[9 MONITOR]   input drift score={drift:.2f}  low-confidence rate={low_conf:.2f}")

    # TEST ---------------------------------------------------------------------
    assert test_acc > 0.90, f"test accuracy too low: {test_acc:.3f}"
    assert served.predict((new_points - smu) / ssd).shape == (3,), "bad inference shape"
    assert drift < 1.0, f"unexpected input drift: {drift:.2f}"
    print(f"[TEST]        PASS  (test_acc {test_acc:.3f} > 0.90, drift {drift:.2f} < 1.0)")


if __name__ == "__main__":
    main()
