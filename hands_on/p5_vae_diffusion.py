"""
Project 5 (core mechanics) -- the VAE reparameterization trick and the DDPM
diffusion equations, in pure NumPy.

The full Project 5 in the book builds a VAE and a DDPM with a U-Net in PyTorch.
The two pieces that are pure math, and that people most need to *see* work, are:

  (A) VAE reparameterization:  z = mu + sigma * eps,  eps ~ N(0, I)
      plus the closed-form KL( N(mu,sigma^2) || N(0,1) ).
  (B) DDPM forward noising (closed form):
      x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
      and a single reverse denoising step given the (here, true) noise.

Run:  python p5_vae_diffusion.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# (A) VAE reparameterization trick + KL term (Chapter 6 losses, generative).
# ---------------------------------------------------------------------------
def reparameterize(mu, logvar, n=100000):
    sigma = np.exp(0.5 * logvar)
    eps = rng.standard_normal((n,) + mu.shape)      # noise sampled OUTSIDE the params
    return mu + sigma * eps                         # differentiable w.r.t. mu, logvar


def kl_standard_normal(mu, logvar):
    # KL( N(mu, sigma^2) || N(0,1) ) = -0.5 * sum(1 + logvar - mu^2 - e^logvar)
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))


def demo_vae():
    print("=== (A) VAE reparameterization trick ===")
    mu = np.array([2.0, -1.0])
    logvar = np.array([np.log(0.25), np.log(1.0)])     # sigmas = 0.5, 1.0
    z = reparameterize(mu, logvar)
    print(f"target mu     = {mu},  target sigma = {np.exp(0.5*logvar)}")
    print(f"sampled mean  = {z.mean(0).round(3)},  sampled std  = {z.std(0).round(3)}")
    print(f"KL(N(mu,s^2)||N(0,1)) = {kl_standard_normal(mu, logvar):.4f}")


# ---------------------------------------------------------------------------
# (B) DDPM forward/reverse (Chapter on diffusion, bonus systems).
# ---------------------------------------------------------------------------
def make_schedule(T=200, beta1=1e-4, betaT=0.02):
    betas = np.linspace(beta1, betaT, T)
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    return betas, alphas, alpha_bar


def demo_diffusion():
    print("\n=== (B) DDPM forward noising (closed form) ===")
    T = 200
    betas, alphas, alpha_bar = make_schedule(T)
    x0 = np.array([1.0])                                # a clean 1-D "signal"
    print("  t     sqrt(abar)   sqrt(1-abar)   Var[x_t]    SNR")
    for t in [0, 49, 99, 149, 199]:
        eps = rng.standard_normal((20000,) + x0.shape)
        xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
        snr = alpha_bar[t] / (1 - alpha_bar[t])
        print(f"  {t:3d}   {np.sqrt(alpha_bar[t]):.4f}      "
              f"{np.sqrt(1-alpha_bar[t]):.4f}        {xt.var():.4f}     {snr:7.3f}")

    # One reverse step using the TRUE noise (sanity check that the math inverts).
    print("\n=== (B) one reverse denoising step (with the true noise) ===")
    t = 100
    eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    # The model predicts eps; here we hand it the true eps to verify the formula.
    x0_hat = (xt - np.sqrt(1 - alpha_bar[t]) * eps) / np.sqrt(alpha_bar[t])
    print(f"  x0 (true)      = {x0}")
    print(f"  x_t (noised)   = {xt.round(4)}")
    print(f"  x0_hat (recon) = {x0_hat.round(4)}   <- recovers x0 from x_t and eps")


if __name__ == "__main__":
    demo_vae()
    demo_diffusion()
