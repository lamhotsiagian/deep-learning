"""
Project 5 -- A trainable Variational Autoencoder in pure NumPy.

A full end-to-end pipeline with a real VAE trained by hand: encoder ->
reparameterization trick -> decoder, with the ELBO (reconstruction BCE + KL)
and every gradient derived manually (Chapter 6 / Chapter 17 generative math):
  1 OBJECTIVE -> 2 DATA -> 3 PREPROCESS -> 4 SPLIT -> 5 MODEL ->
  6 TRAIN(+val) -> 7 TUNE -> 8 EVALUATE -> 9 DEPLOY+MONITOR -> TEST
The hand-derived gradients are verified numerically before training.

(The DDPM diffusion math demo lives in p5_vae_diffusion.py.)

Run:  python p5_vae.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

rng = np.random.default_rng(0)
W = 6                        # image is WxW binary
D = W * W                    # 36 input dims
HENC, HDEC, L = 24, 24, 4    # encoder hidden, decoder hidden, latent dim


def make_bars(n):
    """Binary images, each a vertical bar at a random column (+ bit-flip noise)."""
    X = np.zeros((n, D))
    for i in range(n):
        img = np.zeros((W, W))
        img[:, rng.integers(0, W)] = 1.0                       # vertical bar
        flip = rng.random((W, W)) < 0.04                       # 4% noise
        img = np.abs(img - flip)
        X[i] = img.ravel()
    return X


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def relu(z):
    return np.maximum(0, z)


class VAE:
    def __init__(self):
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.P = {
            "We": s(D, HENC), "be": np.zeros(HENC),
            "Wmu": s(HENC, L), "bmu": np.zeros(L),
            "Wlv": s(HENC, L), "blv": np.zeros(L),
            "Wd1": s(L, HDEC), "bd1": np.zeros(HDEC),
            "Wd2": s(HDEC, D), "bd2": np.zeros(D),
        }

    def forward(self, x, eps):
        P = self.P; c = {}
        c["he_pre"] = x @ P["We"] + P["be"]; c["he"] = relu(c["he_pre"])
        c["mu"] = c["he"] @ P["Wmu"] + P["bmu"]
        c["lv"] = c["he"] @ P["Wlv"] + P["blv"]
        c["std"] = np.exp(0.5 * c["lv"])
        c["z"] = c["mu"] + c["std"] * eps                      # reparameterization
        c["hd_pre"] = c["z"] @ P["Wd1"] + P["bd1"]; c["hd"] = relu(c["hd_pre"])
        c["logits"] = c["hd"] @ P["Wd2"] + P["bd2"]
        c["recon"] = sigmoid(c["logits"])
        c["x"], c["eps"] = x, eps
        return c

    @staticmethod
    def loss(c):
        n = len(c["x"])
        bce = -np.sum(c["x"] * np.log(c["recon"] + 1e-9) +
                      (1 - c["x"]) * np.log(1 - c["recon"] + 1e-9)) / n
        kl = -0.5 * np.sum(1 + c["lv"] - c["mu"] ** 2 - np.exp(c["lv"])) / n
        return bce + kl, bce, kl

    def backward(self, c):
        P = self.P; n = len(c["x"]); g = {}
        dlogits = (c["recon"] - c["x"]) / n                    # BCE+sigmoid grad
        g["Wd2"] = c["hd"].T @ dlogits; g["bd2"] = dlogits.sum(0)
        dhd = (dlogits @ P["Wd2"].T) * (c["hd_pre"] > 0)
        g["Wd1"] = c["z"].T @ dhd; g["bd1"] = dhd.sum(0)
        dz = dhd @ P["Wd1"].T                                  # recon grad wrt z
        # KL grads (per-sample, averaged over batch)
        dmu = dz + c["mu"] / n
        dlv = dz * c["eps"] * 0.5 * c["std"] + 0.5 * (np.exp(c["lv"]) - 1) / n
        g["Wmu"] = c["he"].T @ dmu; g["bmu"] = dmu.sum(0)
        g["Wlv"] = c["he"].T @ dlv; g["blv"] = dlv.sum(0)
        dhe = (dmu @ P["Wmu"].T + dlv @ P["Wlv"].T) * (c["he_pre"] > 0)
        g["We"] = c["x"].T @ dhe; g["be"] = dhe.sum(0)
        return g

    def reconstruct(self, x):
        return self.forward(x, np.zeros((len(x), L))).get("recon")  # eps=0 -> use mean

    def generate(self, n):
        z = rng.standard_normal((n, L))
        hd = relu(z @ self.P["Wd1"] + self.P["bd1"])
        return sigmoid(hd @ self.P["Wd2"] + self.P["bd2"])


def gradient_check(vae, x, eps_fixed, key="Wmu", eps=1e-5):
    c = vae.forward(x, eps_fixed); g = vae.backward(c)
    arr = vae.P[key]; rels = []
    for _ in range(8):
        a, b = rng.integers(arr.shape[0]), rng.integers(arr.shape[1])
        o = arr[a, b]
        arr[a, b] = o + eps; lp = vae.loss(vae.forward(x, eps_fixed))[0]
        arr[a, b] = o - eps; lm = vae.loss(vae.forward(x, eps_fixed))[0]
        arr[a, b] = o
        num = (lp - lm) / (2 * eps)
        rels.append(abs(num - g[key][a, b]) / (abs(num) + abs(g[key][a, b]) + 1e-12))
    return max(rels)


def main():
    print("[1 OBJECTIVE] learn a generative model of binary bar-images; metric=val recon loss")

    # 2) DATA
    X = make_bars(1200)
    print(f"[2 DATA]      generated {len(X)} binary {W}x{W} images (D={D})")

    # 4) SPLIT 70/15/15
    n1, n2 = int(0.70 * len(X)), int(0.85 * len(X))
    Xtr, Xva, Xte = X[:n1], X[n1:n2], X[n2:]
    print(f"[4 SPLIT]     train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # 3) PREPROCESS (already in [0,1]; VAE models Bernoulli pixels directly)
    print("[3 PREPROCESS] pixels are Bernoulli in {0,1}; no scaling needed")

    # 5) MODEL + gradient check (fixed eps so the loss is deterministic)
    vae = VAE()
    rel = max(gradient_check(vae, Xtr[:16], rng.standard_normal((16, L)), k)
              for k in ["Wmu", "Wlv", "We", "Wd1", "Wd2"])
    print(f"[5 MODEL]     VAE enc({D}->{HENC}->{L}) reparam dec({L}->{HDEC}->{D}), Bernoulli")
    print(f"[5 MODEL]     ELBO gradient check max rel err {rel:.2e} -> {'PASS' if rel < 1e-4 else 'FAIL'}")

    # 6) TRAIN + 7) TUNE (best-val ELBO checkpoint)
    state = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v)} for k, v in vae.P.items()}
    lr, b1, b2, t = 3e-3, 0.9, 0.999, 0
    best_val, best = np.inf, None
    for epoch in range(1, 61):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            xb = Xtr[order[s:s + 64]]
            c = vae.forward(xb, rng.standard_normal((len(xb), L)))
            g = vae.backward(c); t += 1
            for k in vae.P:
                state[k]["m"] = b1 * state[k]["m"] + (1 - b1) * g[k]
                state[k]["v"] = b2 * state[k]["v"] + (1 - b2) * g[k] ** 2
                mh = state[k]["m"] / (1 - b1 ** t); vh = state[k]["v"] / (1 - b2 ** t)
                vae.P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        vloss = vae.loss(vae.forward(Xva, np.zeros((len(Xva), L))))[0]
        if vloss < best_val:
            best_val, best = vloss, {k: v.copy() for k, v in vae.P.items()}
        if epoch % 15 == 0:
            tl, bce, kl = vae.loss(vae.forward(Xtr, np.zeros((len(Xtr), L))))
            print(f"[6 TRAIN]     epoch {epoch:2d}  ELBO {tl:.3f} (bce {bce:.3f} + kl {kl:.3f})  val {vloss:.3f}")
    vae.P = best
    print(f"[7 TUNE]      restored best-val checkpoint (val ELBO {best_val:.3f})")

    # 8) EVALUATE: reconstruction loss + per-pixel accuracy on test
    rec = vae.reconstruct(Xte)
    test_elbo = vae.loss(vae.forward(Xte, np.zeros((len(Xte), L))))[0]
    pix_acc = ((rec > 0.5) == (Xte > 0.5)).mean()
    print(f"[8 EVALUATE]  test_ELBO={test_elbo:.3f}  pixel_recon_acc={pix_acc:.3f}")

    # 9) DEPLOY + INFERENCE + MONITOR
    tmp = tempfile.mkdtemp(); path = os.path.join(tmp, "vae.npz")
    np.savez(path, **vae.P)
    z = np.load(path); served = VAE(); served.P = {k: z[k] for k in vae.P}
    gen = served.generate(8)                                  # sample new images from N(0,I)
    bars_per_gen = (gen.reshape(8, W, W) > 0.5).sum(axis=(1, 2)).mean()
    print(f"[9 DEPLOY]    saved+reloaded; generated 8 new images (avg ~{bars_per_gen:.0f} on-pixels each)")
    train_elbo = vae.loss(vae.forward(Xtr[:len(Xte)], np.zeros((len(Xte), L))))[0]
    print(f"[9 MONITOR]   train_ELBO={train_elbo:.3f}  test_ELBO={test_elbo:.3f}  gap={test_elbo - train_elbo:+.3f}")

    # TEST
    assert rel < 1e-4, f"gradient check failed: {rel:.2e}"
    assert pix_acc > 0.90, f"reconstruction accuracy too low: {pix_acc:.3f}"
    print(f"[TEST]        PASS  (grad-check {rel:.1e}, pixel_recon_acc {pix_acc:.3f} > 0.90)")


if __name__ == "__main__":
    main()
