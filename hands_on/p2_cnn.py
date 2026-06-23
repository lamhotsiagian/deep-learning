"""
Project 2 -- A trainable Convolutional Neural Network in pure NumPy.

A full end-to-end pipeline with a real conv layer trained by hand (im2col
forward + backward, Chapter 15). It classifies 8x8 images of oriented lines
(vertical / horizontal / diagonal):
  1 OBJECTIVE -> 2 DATA -> 3 PREPROCESS -> 4 SPLIT -> 5 MODEL ->
  6 TRAIN(+val) -> 7 TUNE -> 8 EVALUATE -> 9 DEPLOY+MONITOR -> TEST
The hand-derived conv gradient is verified numerically before training.

(The production ResNet/CIFAR-10 version is the PyTorch code in the chapter and
in p2_resnet_shapes.py.)

Run:  python p2_cnn.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)
IMG = 8                      # image size
K, PAD = 3, 1               # 3x3 conv, 'same' padding
COUT = 6                    # number of conv filters
NCLS = 3                    # vertical / horizontal / diagonal


# ---------------------------------------------------------------------------
# 2-3) DATA + PREPROCESS: synthetic oriented-line images
# ---------------------------------------------------------------------------
def make_images(n):
    X = np.zeros((n, 1, IMG, IMG)); y = rng.integers(0, NCLS, n)
    for i in range(n):
        img = np.zeros((IMG, IMG))
        off = rng.integers(1, IMG - 1)
        if y[i] == 0:                      # vertical line
            img[:, off] = 1.0
        elif y[i] == 1:                    # horizontal line
            img[off, :] = 1.0
        else:                              # diagonal line
            for d in range(IMG):
                img[d, d] = 1.0
        img += rng.normal(0, 0.15, (IMG, IMG))    # noise (augmentation)
        X[i, 0] = img
    return X, y


def im2col(X):
    """X:(N,C,H,W) -> cols:(N, C*K*K, H*W), stride 1, 'same' pad."""
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (PAD, PAD), (PAD, PAD)))
    cols = np.empty((N, C * K * K, H * W)); idx = 0
    for c in range(C):
        for i in range(K):
            for j in range(K):
                cols[:, idx, :] = Xp[:, c, i:i + H, j:j + W].reshape(N, H * W)
                idx += 1
    return cols


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class ConvNet:
    """conv(1->COUT,3x3) -> ReLU -> global avg pool -> linear -> softmax."""

    def __init__(self):
        self.Wc = rng.standard_normal((COUT, 1 * K * K)) * np.sqrt(2.0 / (K * K))
        self.bc = np.zeros(COUT)
        self.Wf = rng.standard_normal((COUT, NCLS)) * np.sqrt(2.0 / COUT)
        self.bf = np.zeros(NCLS)

    def forward(self, X):
        N, _, H, W = X.shape
        self.cols = im2col(X)                                      # (N, K*K, HW)
        conv = np.einsum("oc,ncp->nop", self.Wc, self.cols) + self.bc[None, :, None]
        self.conv = conv.reshape(N, COUT, H, W)
        self.relu = np.maximum(0, self.conv)
        self.pooled = self.relu.mean(axis=(2, 3))                  # global avg pool (N, COUT)
        self.logits = self.pooled @ self.Wf + self.bf
        return self.logits

    def backward(self, X, y):
        N, _, H, W = X.shape
        probs = softmax(self.logits)
        dlogits = probs.copy(); dlogits[np.arange(N), y] -= 1; dlogits /= N
        dWf = self.pooled.T @ dlogits; dbf = dlogits.sum(0)
        dpooled = dlogits @ self.Wf.T                             # (N, COUT)
        drelu = np.broadcast_to((dpooled / (H * W))[:, :, None, None], self.relu.shape)
        dconv = (drelu * (self.conv > 0)).reshape(N, COUT, H * W)
        dWc = np.einsum("nop,ncp->oc", dconv, self.cols)          # (COUT, K*K)
        dbc = dconv.sum(axis=(0, 2))
        return dWc, dbc, dWf, dbf

    def predict(self, X):
        return self.forward(X).argmax(1)


def ce(net, X, y):
    return -np.log(softmax(net.forward(X))[np.arange(len(y)), y] + 1e-12).mean()


def gradient_check(net, X, y, eps=1e-5):
    net.forward(X)
    dWc, dbc, dWf, dbf = net.backward(X, y)
    rels = []
    for arr, grad in [(net.Wc, dWc), (net.Wf, dWf)]:
        for _ in range(6):
            a, b = rng.integers(arr.shape[0]), rng.integers(arr.shape[1])
            o = arr[a, b]
            arr[a, b] = o + eps; lp = ce(net, X, y)
            arr[a, b] = o - eps; lm = ce(net, X, y); arr[a, b] = o
            num = (lp - lm) / (2 * eps)
            rels.append(abs(num - grad[a, b]) / (abs(num) + abs(grad[a, b]) + 1e-12))
    return max(rels)


def main():
    print("[1 OBJECTIVE] classify 8x8 oriented-line images (V/H/diagonal); metric=test acc")

    # 2) DATA
    X, y = make_images(900)
    print(f"[2 DATA]      generated {len(y)} images, {NCLS} classes, shape {X.shape[1:]}")

    # 4) SPLIT 70/15/15
    n1, n2 = int(0.70 * len(y)), int(0.85 * len(y))
    Xtr, ytr, Xva, yva, Xte, yte = X[:n1], y[:n1], X[n1:n2], y[n1:n2], X[n2:], y[n2:]
    print(f"[4 SPLIT]     train={len(ytr)}  val={len(yva)}  test={len(yte)}")

    # 3) PREPROCESS: standardize pixels with train stats
    mu, sd = Xtr.mean(), Xtr.std() + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"[3 PREPROCESS] pixel standardization (mu={mu:.3f}, sd={sd:.3f})")

    # 5) MODEL + gradient check
    net = ConvNet()
    rel = gradient_check(net, Xtr[:16], ytr[:16])
    print(f"[5 MODEL]     conv(1->{COUT},3x3)->ReLU->GAP->linear->softmax")
    print(f"[5 MODEL]     conv/linear gradient check max rel err {rel:.2e} -> {'PASS' if rel < 1e-4 else 'FAIL'}")

    # 6) TRAIN (mini-batch) + 7) TUNE (best-val checkpoint)
    mWc = mbc = mWf = mbf = None
    state = {"mWc": np.zeros_like(net.Wc), "vWc": np.zeros_like(net.Wc),
             "mbc": np.zeros_like(net.bc), "vbc": np.zeros_like(net.bc),
             "mWf": np.zeros_like(net.Wf), "vWf": np.zeros_like(net.Wf),
             "mbf": np.zeros_like(net.bf), "vbf": np.zeros_like(net.bf)}
    lr, b1, b2, eps, t = 5e-3, 0.9, 0.999, 1e-8, 0
    best_va, best = 0.0, None
    for epoch in range(1, 41):
        order = rng.permutation(len(ytr))
        for s in range(0, len(ytr), 32):
            bi = order[s:s + 32]
            net.forward(Xtr[bi])
            gWc, gbc, gWf, gbf = net.backward(Xtr[bi], ytr[bi]); t += 1
            for name, param, g in [("Wc", net.Wc, gWc), ("bc", net.bc, gbc),
                                   ("Wf", net.Wf, gWf), ("bf", net.bf, gbf)]:
                state["m" + name] = b1 * state["m" + name] + (1 - b1) * g
                state["v" + name] = b2 * state["v" + name] + (1 - b2) * g ** 2
                mh = state["m" + name] / (1 - b1 ** t); vh = state["v" + name] / (1 - b2 ** t)
                param -= lr * mh / (np.sqrt(vh) + eps)
        va = (net.predict(Xva) == yva).mean()
        if va > best_va:
            best_va, best = va, (net.Wc.copy(), net.bc.copy(), net.Wf.copy(), net.bf.copy())
        if epoch % 10 == 0:
            print(f"[6 TRAIN]     epoch {epoch:2d}  train_loss {ce(net, Xtr, ytr):.4f}  val_acc {va:.3f}")
    net.Wc, net.bc, net.Wf, net.bf = best
    print(f"[7 TUNE]      restored best-val checkpoint (val_acc {best_va:.3f})")

    # 8) EVALUATE
    test_acc = (net.predict(Xte) == yte).mean()
    print(f"[8 EVALUATE]  test_acc={test_acc:.3f}")

    # 9) DEPLOY + INFERENCE + MONITOR
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "cnn.npz")
    np.savez(path, Wc=net.Wc, bc=net.bc, Wf=net.Wf, bf=net.bf)
    z = np.load(path); served = ConvNet()
    served.Wc, served.bc, served.Wf, served.bf = z["Wc"], z["bc"], z["Wf"], z["bf"]
    Xnew, ynew = make_images(20); Xnew = (Xnew - mu) / sd
    proba = softmax(served.forward(Xnew)); preds, conf = proba.argmax(1), proba.max(1)
    acc_new = (preds == ynew).mean()
    print(f"[9 DEPLOY]    saved+reloaded; inference on 20 new images, first 5 -> {preds[:5].tolist()} (true {ynew[:5].tolist()})")
    print(f"[9 MONITOR]   live-batch accuracy={acc_new:.2f}  mean confidence={conf.mean():.2f}")

    # TEST
    assert rel < 1e-4, f"gradient check failed: {rel:.2e}"
    assert test_acc > 0.90, f"test accuracy too low: {test_acc:.3f}"
    assert acc_new >= 0.85, f"live-batch accuracy too low: {acc_new:.2f}"
    print(f"[TEST]        PASS  (grad-check {rel:.1e}, test_acc {test_acc:.3f} > 0.90, live_acc {acc_new:.2f})")


if __name__ == "__main__":
    main()
