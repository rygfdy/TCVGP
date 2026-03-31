# ablation_vigp_caseb_r_h.py
# Requirements: python>=3.9, torch, numpy, matplotlib
# Run: python ablation_vigp_caseb_r_h.py

import math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# -----------------------------
# Case (b) data
# -----------------------------
def f_true(x):
    return 2.0 * torch.log1p(x)

def sample_case_b(n=30, x_min=0.0, x_max=10.0, noise_std=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (x_max - x_min) * torch.rand(n, generator=g) + x_min
    x, _ = torch.sort(x)
    y = f_true(x) + noise_std * torch.randn(n, generator=g)
    return x, y

# -----------------------------
# RBF kernel and derivative covariances (1D)
# k(x,x') = sf2 * exp(-(x-x')^2/(2*l^2))
# Cov(f(x), f(x')) = k
# Cov(f(x), f'(z)) = d/dz k(x,z) = k(x,z) * (x - z)/l^2
# Cov(f'(z), f'(z')) = d/dz d/dz' k(z,z')
#                   = k(z,z') * (1/l^2 - (z-z')^2/l^4)
# -----------------------------
def rbf_kernel(x, z, sf2, ell):
    x = x[:, None]
    z = z[None, :]
    r2 = (x - z) ** 2
    return sf2 * torch.exp(-0.5 * r2 / (ell ** 2))

def cov_f_f(X, sf2, ell):
    return rbf_kernel(X, X, sf2, ell)

def cov_f_d(X, Z, sf2, ell):
    # Cov(f(X_i), f'(Z_j))
    Kxz = rbf_kernel(X, Z, sf2, ell)
    diff = (X[:, None] - Z[None, :])
    return Kxz * (diff / (ell ** 2))

def cov_d_d(Z, sf2, ell):
    Kzz = rbf_kernel(Z, Z, sf2, ell)
    diff = (Z[:, None] - Z[None, :])
    return Kzz * (1.0 / (ell ** 2) - (diff ** 2) / (ell ** 4))

def build_joint_cov(X, Z, sf2, ell, jitter=1e-6):
    Kff = cov_f_f(X, sf2, ell)
    Kfd = cov_f_d(X, Z, sf2, ell)
    Kdd = cov_d_d(Z, sf2, ell)
    top = torch.cat([Kff, Kfd], dim=1)
    bot = torch.cat([Kfd.t(), Kdd], dim=1)
    Sigma = torch.cat([top, bot], dim=0)
    Sigma = Sigma + jitter * torch.eye(Sigma.shape[0], dtype=Sigma.dtype)
    return Sigma

# -----------------------------
# Positive monotone maps h: R -> R_+
# Provide: h(u), log h'(u)
# -----------------------------
class HMap:
    def __init__(self, name, beta=1.0, eps=1e-9):
        self.name = name
        self.beta = float(beta)
        self.eps = float(eps)

    def h(self, u):
        if self.name == "softplus":
            return torch.nn.functional.softplus(self.beta * u) / self.beta + self.eps
        if self.name == "exp":
            # can be aggressive; add clamp for stability
            return torch.exp(torch.clamp(u, -50.0, 50.0)) + self.eps
        if self.name == "squareplus":
            # squareplus: 0.5*(u + sqrt(u^2 + b))
            b = 1.0
            return 0.5 * (u + torch.sqrt(u * u + b)) + self.eps
        raise ValueError(f"Unknown h: {self.name}")

    def log_hprime(self, u):
        if self.name == "softplus":
            # h'(u) = sigmoid(beta*u)
            return torch.log(torch.sigmoid(self.beta * u) + self.eps)
        if self.name == "exp":
            # h'(u)=exp(u)
            return torch.clamp(u, -50.0, 50.0)
        if self.name == "squareplus":
            b = 1.0
            # derivative: 0.5*(1 + u/sqrt(u^2+b))
            denom = torch.sqrt(u * u + b)
            return torch.log(0.5 * (1.0 + u / denom) + self.eps)
        raise ValueError(f"Unknown h: {self.name}")

# -----------------------------
# VIGP variational family (square A with low-rank structure)
# g = [f; d]
# f = mu_f + L_f z_f, z_f~N(0,I_mf)
# u = mu_d + A z_d, z_d~N(0,I_md)
# d = h(u) (enforces d>=0)
#
# A = diag(exp(s)) + U V^T , with U,V in R^{md x r}
# log|A| via determinant lemma:
# log|D + U V^T| = sum log D_i + log|I_r + V^T D^{-1} U|
# -----------------------------
class VIGP(nn.Module):
    def __init__(self, mf, md, rank_r, hmap: HMap):
        super().__init__()
        self.mf = mf
        self.md = md
        self.r = int(rank_r)
        self.hmap = hmap

        # f-part
        self.mu_f = nn.Parameter(torch.zeros(mf))
        # Cholesky parameter (lower tri)
        self.L_f_uncon = nn.Parameter(torch.eye(mf))

        # d-part
        self.mu_d = nn.Parameter(torch.zeros(md))
        self.log_diag_D = nn.Parameter(torch.zeros(md))  # D = diag(exp(log_diag_D)) positive
        if self.r > 0:
            self.U = nn.Parameter(0.01 * torch.randn(md, self.r))
            self.V = nn.Parameter(0.01 * torch.randn(md, self.r))
        else:
            self.U = None
            self.V = None

    def L_f(self):
        # force lower-triangular with positive diag
        L = torch.tril(self.L_f_uncon)
        diag = torch.diagonal(L)
        L = L - torch.diag(diag) + torch.diag(torch.exp(diag))
        return L

    def A(self):
        D = torch.diag(torch.exp(self.log_diag_D))
        if self.r > 0:
            return D + self.U @ self.V.t()
        else:
            return D

    def logdet_A(self):
        # log|D + U V^T|
        diagD = torch.exp(self.log_diag_D)  # (md,)
        logdetD = torch.sum(torch.log(diagD))
        if self.r == 0:
            return logdetD
        # M = I_r + V^T D^{-1} U
        Dinvu = self.U / diagD[:, None]
        M = torch.eye(self.r, dtype=diagD.dtype) + self.V.t() @ Dinvu
        sign, logabsdetM = torch.slogdet(M)
        # with our parameterization it should be positive; still guard
        return logdetD + logabsdetM

    def sample_g_and_logq(self, n_samples=64):
        mf, md = self.mf, self.md
        zf = torch.randn(n_samples, mf)
        zd = torch.randn(n_samples, md)

        Lf = self.L_f()
        f = self.mu_f[None, :] + zf @ Lf.t()

        A = self.A()
        u = self.mu_d[None, :] + zd @ A.t()
        d = self.hmap.h(u)

        g = torch.cat([f, d], dim=1)

        # log q(g) via change of variables from (zf,zd) -> (f,u) -> (f,d)
        # log q = log N(zf)+log N(zd) - log|det Lf| - log|det A| - sum log h'(u)
        logN_zf = -0.5 * (zf**2).sum(dim=1) - 0.5 * mf * math.log(2 * math.pi)
        logN_zd = -0.5 * (zd**2).sum(dim=1) - 0.5 * md * math.log(2 * math.pi)

        logdetLf = torch.sum(torch.log(torch.diagonal(Lf)))
        logdetA = self.logdet_A()
        logdetJh = self.hmap.log_hprime(u).sum(dim=1)

        logq = (logN_zf + logN_zd) - (logdetLf + logdetA) - logdetJh
        return g, logq

# -----------------------------
# ELBO for hard posterior (up to constant):
#   E_q[ log p(y|f) + log N(g;0,Sigma) - log q(g) ]
# where p(y|f)=N(y|f, sigma_n^2 I)
# -----------------------------
def elbo(model: VIGP, y, Sigma, noise_std, n_mc=64):
    g, logq = model.sample_g_and_logq(n_samples=n_mc)
    mf = model.mf
    f = g[:, :mf]  # (n_mc, mf)

    # log p(y|f)
    resid = y[None, :] - f
    loglik = -0.5 * (resid**2).sum(dim=1) / (noise_std**2) - 0.5 * mf * math.log(2 * math.pi * (noise_std**2))

    # log prior N(g;0,Sigma)
    # use Cholesky for stability
    L = torch.linalg.cholesky(Sigma)
    # solve Sigma^{-1} g^T via triangular solves
    # For each sample: quad = g Sigma^{-1} g^T
    v = torch.cholesky_solve(g.t(), L)  # (m, n_mc)
    quad = (g.t() * v).sum(dim=0)       # (n_mc,)
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    m = Sigma.shape[0]
    logprior = -0.5 * quad - 0.5 * logdet - 0.5 * m * math.log(2 * math.pi)

    return (loglik + logprior - logq).mean()

# -----------------------------
# Prediction: use Monte Carlo moments of g then propagate (paper eq 29-31) :contentReference[oaicite:2]{index=2}
# -----------------------------
def predict_from_q_moments(X_train, Z, X_test, Sigma, sf2, ell, model: VIGP, n_mc=2000,
                           cov_shrink=1e-3, jitter=1e-8):
    """
    Stable moment-based prediction:
      m_q = E_q[g], C_q = Cov_q[g]
      f* | g is Gaussian, propagate moments without explicitly inverting Sigma.

    Key stabilizations:
      - torch covariance with shrinkage (PSD-ish)
      - cholesky_solve instead of explicit inverse
      - small jitter on cov diagonals
    """
    with torch.no_grad():
        g_s, _ = model.sample_g_and_logq(n_samples=n_mc)  # (n_mc, m)
        m_q = g_s.mean(dim=0)                             # (m,)
        Xc = g_s - m_q[None, :]
        C_q = (Xc.t() @ Xc) / (n_mc - 1)                  # (m,m)

        # shrinkage towards diagonal to reduce MC noise
        diag = torch.diagonal(C_q)
        C_q = (1.0 - cov_shrink) * C_q + cov_shrink * torch.diag(diag)

        # jitter to avoid numerical issues
        m = C_q.shape[0]
        C_q = C_q + jitter * torch.eye(m, dtype=C_q.dtype)

    # Cross-cov K_g* = Cov(g, f*)
    K_f_star = rbf_kernel(X_train, X_test, sf2, ell)      # (mf, nt)
    K_z_star = rbf_kernel(Z, X_test, sf2, ell)            # (md, nt)
    diff = (X_test[None, :] - Z[:, None])                 # (md, nt)
    K_d_star = K_z_star * (diff / (ell**2))               # (md, nt)

    K_g_star = torch.cat([K_f_star, K_d_star], dim=0)     # (m, nt)
    K_star_star = rbf_kernel(X_test, X_test, sf2, ell)    # (nt, nt)

    # Solve systems with Sigma using Cholesky
    L = torch.linalg.cholesky(Sigma)

    def solveSigma(B):
        # returns Sigma^{-1} B
        return torch.cholesky_solve(B, L)

    # mean: K_*g Sigma^{-1} m_q
    Sinv_m = solveSigma(m_q[:, None])                     # (m,1)
    mean = (K_g_star.t() @ Sinv_m).squeeze(1)            # (nt,)

    # base cov: K** - K_*g Sigma^{-1} K_g*
    Sinv_K = solveSigma(K_g_star)                         # (m,nt)
    base = K_star_star - K_g_star.t() @ Sinv_K            # (nt,nt)

    # correction: K_*g Sigma^{-1} C_q Sigma^{-1} K_g*
    # compute Sigma^{-1} C_q Sigma^{-1} K efficiently
    # Step1: T = Sigma^{-1} K
    # Step2: U = C_q T
    # Step3: V = Sigma^{-1} U
    U = C_q @ Sinv_K                                      # (m,nt)
    V = solveSigma(U)                                     # (m,nt)
    corr = K_g_star.t() @ V                               # (nt,nt)

    cov = base + corr
    cov = cov + jitter * torch.eye(cov.shape[0], dtype=cov.dtype)

    var = torch.diagonal(cov).clamp_min(1e-10)
    return mean, var

# -----------------------------
# Ablation runner
# -----------------------------
def run_one_setting(
    X, y, Z, X_test, noise_std,
    sf2=10.0, ell=1.5,
    rank_r=2,
    h_name="softplus",
    steps=2000,
    lr=2e-2,
    mc_train=64,
    mc_pred=512,
    seed=0,
    verbose=False
):
    torch.manual_seed(seed)
    mf, md = X.numel(), Z.numel()
    Sigma = build_joint_cov(X, Z, sf2=sf2, ell=ell, jitter=1e-6)

    hmap = HMap(h_name, beta=1.0)
    model = VIGP(mf=mf, md=md, rank_r=rank_r, hmap=hmap)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        L = elbo(model, y, Sigma, noise_std=noise_std, n_mc=mc_train)
        loss = -L
        loss.backward()
        opt.step()

        if verbose and (it % 200 == 0 or it == steps - 1):
            print(f"[r={rank_r:2d} h={h_name:9s}] it={it:4d} ELBO={L.item():.3f}")

    train_time = time.time() - t0

    # Predict
    mean, var = predict_from_q_moments(X, Z, X_test, Sigma, sf2, ell, model, n_mc=mc_pred)
    y_true = f_true(X_test)

    rmse = torch.sqrt(torch.mean((mean - y_true) ** 2)).item()
    nll = (0.5 * torch.log(2 * math.pi * var) + 0.5 * ((y_true - mean) ** 2) / var).mean().item()

    # Constraint violation prob at Z under q: should be ~0 by construction
    with torch.no_grad():
        g_s, _ = model.sample_g_and_logq(n_samples=2000)
        d_s = g_s[:, mf:]
        viol = (d_s < 0).any(dim=1).double().mean().item()

    return {
        "r": rank_r,
        "h": h_name,
        "md": md,
        "RMSE": rmse,
        "NLL": nll,
        "viol_prob": viol,
        "train_sec": train_time,
        "model": model,
        "Sigma": Sigma,
        "pred_mean": mean.detach().cpu(),
        "pred_var": var.detach().cpu(),
        "X_test": X_test.detach().cpu(),
    }


def run_md_ablation(
    X,
    y,
    X_test,
    noise_std,
    md_list,
    sf2=10.0,
    ell=1.5,
    rank_r=2,
    h_name="softplus",
    steps=1500,
    lr=2e-2,
    mc_train=64,
    mc_pred=512,
    seed=123,
):
    results = []
    for md in md_list:
        Z = torch.linspace(float(X.min()), float(X.max()), int(md), dtype=X.dtype)
        out = run_one_setting(
            X,
            y,
            Z,
            X_test,
            noise_std,
            sf2=sf2,
            ell=ell,
            rank_r=rank_r,
            h_name=h_name,
            steps=steps,
            lr=lr,
            mc_train=mc_train,
            mc_pred=mc_pred,
            seed=seed,
            verbose=False,
        )
        results.append(out)
    return results


def run_mf_ablation(
    mf_list,
    X_test,
    noise_std,
    md=8,
    sf2=10.0,
    ell=1.5,
    rank_r=2,
    h_name="softplus",
    steps=1500,
    lr=2e-2,
    mc_train=64,
    mc_pred=512,
    seed=0,
):
    max_mf = int(max(mf_list))
    X_full, y_full = sample_case_b(n=max_mf, noise_std=noise_std, seed=seed)
    Z = torch.linspace(float(X_full.min()), float(X_full.max()), int(md), dtype=X_full.dtype)

    results = []
    for mf in mf_list:
        mf = int(mf)
        if mf <= 0 or mf > max_mf:
            raise ValueError(f"mf must be in [1, {max_mf}], got {mf}")

        # Use evenly spread indices from one master dataset so the ablation mainly
        # reflects sample size rather than a different draw each time.
        idx = torch.linspace(0, max_mf - 1, mf, dtype=torch.float64).round().long()
        idx = torch.unique_consecutive(idx)
        X = X_full[idx]
        y = y_full[idx]

        out = run_one_setting(
            X,
            y,
            Z,
            X_test,
            noise_std,
            sf2=sf2,
            ell=ell,
            rank_r=rank_r,
            h_name=h_name,
            steps=steps,
            lr=lr,
            mc_train=mc_train,
            mc_pred=mc_pred,
            seed=seed,
            verbose=False,
        )
        out["mf"] = int(X.numel())
        results.append(out)
    return results

class VIGPFull(nn.Module):
    """
    Full transport baseline:
      u = mu + B z ,   z ~ N(0, I_m), m = mf+md
      f = u[:mf]                     (unconstrained)
      d = h(u[mf:])                  (enforce d>=0)
    where B = diag(exp(s)) + U V^T  (low-rank + diagonal)
    """
    def __init__(self, mf, md, rank_r, hmap: HMap):
        super().__init__()
        self.mf = mf
        self.md = md
        self.m = mf + md
        self.r = int(rank_r)
        self.hmap = hmap

        self.mu = nn.Parameter(torch.zeros(self.m))
        self.log_diag_D = nn.Parameter(torch.zeros(self.m))  # D positive diag

        if self.r > 0:
            self.U = nn.Parameter(0.01 * torch.randn(self.m, self.r))
            self.V = nn.Parameter(0.01 * torch.randn(self.m, self.r))
        else:
            self.U = None
            self.V = None

    def B(self):
        D = torch.diag(torch.exp(self.log_diag_D))
        if self.r > 0:
            return D + self.U @ self.V.t()
        else:
            return D

    def logdet_B(self):
        diagD = torch.exp(self.log_diag_D)  # (m,)
        logdetD = torch.sum(torch.log(diagD))
        if self.r == 0:
            return logdetD
        # log|D + U V^T| = log|D| + log|I + V^T D^{-1} U|
        Dinvu = self.U / diagD[:, None]
        M = torch.eye(self.r, dtype=diagD.dtype) + self.V.t() @ Dinvu
        sign, logabsdetM = torch.slogdet(M)
        return logdetD + logabsdetM

    def sample_g_and_logq(self, n_samples=64):
        mf, md, m = self.mf, self.md, self.m
        z = torch.randn(n_samples, m)

        B = self.B()
        u = self.mu[None, :] + z @ B.t()

        f = u[:, :mf]
        u_d = u[:, mf:]
        d = self.hmap.h(u_d)

        g = torch.cat([f, d], dim=1)

        # log q(g): z -> u affine, u_d -> d through monotone map h
        logN_z = -0.5 * (z**2).sum(dim=1) - 0.5 * m * math.log(2 * math.pi)

        logdetB = self.logdet_B()
        logdetJh = self.hmap.log_hprime(u_d).sum(dim=1)

        logq = logN_z - logdetB - logdetJh
        return g, logq


def run_partial_vs_full(
    X, y, Z, X_test,
    noise_std,
    sf2=2.0,
    ell=1.5,
    r=2,
    h_name="softplus",
    steps=1500,
    lr=2e-2,
    mc_train=64,
    mc_pred=3000,
    seed=123,
):
    torch.manual_seed(seed)
    mf, md = X.numel(), Z.numel()
    Sigma = build_joint_cov(X, Z, sf2=sf2, ell=ell, jitter=1e-6)

    hmap = HMap(h_name, beta=1.0)

    # Partial (ours)
    partial = VIGP(mf=mf, md=md, rank_r=r, hmap=hmap)
    opt_p = torch.optim.Adam(partial.parameters(), lr=lr)

    for it in range(steps):
        opt_p.zero_grad(set_to_none=True)
        L = elbo(partial, y, Sigma, noise_std=noise_std, n_mc=mc_train)
        (-L).backward()
        opt_p.step()

    # Full transport baseline
    full = VIGPFull(mf=mf, md=md, rank_r=r, hmap=hmap)
    opt_f = torch.optim.Adam(full.parameters(), lr=lr)

    for it in range(steps):
        opt_f.zero_grad(set_to_none=True)
        L = elbo(full, y, Sigma, noise_std=noise_std, n_mc=mc_train)
        (-L).backward()
        opt_f.step()

    # Predict both
    mu_p, var_p = predict_from_q_moments(X, Z, X_test, Sigma, sf2, ell, partial, n_mc=mc_pred)
    mu_f, var_f = predict_from_q_moments(X, Z, X_test, Sigma, sf2, ell, full, n_mc=mc_pred)
    y_true = f_true(X_test)

    rmse_p = torch.sqrt(torch.mean((mu_p - y_true) ** 2)).item()
    nll_p = (0.5 * torch.log(2 * math.pi * var_p) + 0.5 * ((y_true - mu_p) ** 2) / var_p).mean().item()
    rmse_f = torch.sqrt(torch.mean((mu_f - y_true) ** 2)).item()
    nll_f = (0.5 * torch.log(2 * math.pi * var_f) + 0.5 * ((y_true - mu_f) ** 2) / var_f).mean().item()

    # Check violations
    with torch.no_grad():
        gs, _ = partial.sample_g_and_logq(n_samples=2000)
        viol_p = (gs[:, mf:] < 0).any(dim=1).double().mean().item()
        gs, _ = full.sample_g_and_logq(n_samples=2000)
        viol_f = (gs[:, mf:] < 0).any(dim=1).double().mean().item()

    print(f"[Partial] r={r} h={h_name} | RMSE={rmse_p:.4f} NLL={nll_p:.4f} viol={viol_p:.2e}")
    print(f"[Full   ] r={r} h={h_name} | RMSE={rmse_f:.4f} NLL={nll_f:.4f} viol={viol_f:.2e}")

    # Plot predictive mean + CI
    Xt = X_test.detach().cpu()
    yt = y_true.detach().cpu()

    sp = torch.sqrt(var_p).detach().cpu()
    sfull = torch.sqrt(var_f).detach().cpu()
    mup = mu_p.detach().cpu()
    muf = mu_f.detach().cpu()

    plt.figure()
    plt.scatter(X.detach().cpu().numpy(), y.detach().cpu().numpy(), s=18, alpha=0.8, label="noisy data")
    plt.plot(Xt.numpy(), yt.numpy(), linewidth=2, label="true f(x)")

    # Unconstrained GPR baseline
    mu_gp, var_gp = gpr_predict(X, y, X_test, sf2=10.0, ell=1.5, noise_std=noise_std)
    sgp = torch.sqrt(var_gp).detach().cpu()
    mugp = mu_gp.detach().cpu()

    plt.plot(Xt.numpy(), mugp.numpy(), linewidth=2, label="partial transport")
    plt.fill_between(Xt.numpy(),
                     (mugp - 1.96*sp).numpy(),
                     (mugp + 1.96*sp).numpy(),
                     alpha=0.15)

    # plt.plot(Xt.numpy(), mup.numpy(), linewidth=2, label="partial transport (ours)")
    # plt.fill_between(Xt.numpy(), (mup - 1.96*sp).numpy(), (mup + 1.96*sp).numpy(), alpha=0.2)

    plt.plot(Xt.numpy(), muf.numpy(), linewidth=2, label="full transport")
    plt.fill_between(Xt.numpy(), (muf - 1.96*sfull).numpy(), (muf + 1.96*sfull).numpy(), alpha=0.2)

    plt.title(f"Partial vs Full transport (Case b): r={r}, h={h_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def gpr_predict(X_train, y_train, X_test, sf2, ell, noise_std, jitter=1e-8):
    """
    Standard unconstrained GP regression with RBF kernel.
    Returns posterior mean and variance at X_test.
    """
    K = rbf_kernel(X_train, X_train, sf2, ell)
    K = K + (noise_std**2 + jitter) * torch.eye(X_train.numel(), dtype=K.dtype)

    Ks = rbf_kernel(X_train, X_test, sf2, ell)          # (n, nt)
    Kss = rbf_kernel(X_test, X_test, sf2, ell)          # (nt, nt)

    L = torch.linalg.cholesky(K)

    # alpha = K^{-1} y via solves
    alpha = torch.cholesky_solve(y_train[:, None], L)    # (n,1)

    mean = (Ks.t() @ alpha).squeeze(1)                   # (nt,)

    # v = L^{-1} Ks
    v = torch.linalg.solve_triangular(L, Ks, upper=False)  # (n, nt)
    cov = Kss - v.t() @ v
    var = torch.diagonal(cov).clamp_min(1e-10)
    return mean, var


def main():
    # Data / setup
    noise_std = 1.0
    X, y = sample_case_b(n=30, noise_std=noise_std, seed=0)

    # Constraint points (md small as in paper examples)
    md = 8
    Z = torch.linspace(0.0, 10.0, md)

    X_test = torch.linspace(0.0, 10.0, 200)

    # Ablations
    r_list = [0, 1, 2, 4, 8]          # rank choices
    h_list = ["softplus", "exp", "squareplus"]

    results = []
    best = None

    for h in h_list:
        for r in r_list:
            out = run_one_setting(
                X, y, Z, X_test, noise_std,
                sf2=10.0, ell=1.5,
                rank_r=r, h_name=h,
                steps=1500, lr=2e-2,
                mc_train=64, mc_pred=512,
                seed=123,
                verbose=False
            )
            results.append(out)
            if best is None or out["NLL"] < best["NLL"]:
                best = out
            print(f"r={r:2d} h={h:9s} | RMSE={out['RMSE']:.4f} NLL={out['NLL']:.4f} "
                  f"viol={out['viol_prob']:.4e} time={out['train_sec']:.1f}s")

    # Table-like print
    print("\n=== Summary (sorted by NLL) ===")
    results_sorted = sorted(results, key=lambda d: d["NLL"])
    for d in results_sorted:
        print(f"r={d['r']:2d} h={d['h']:9s}  RMSE={d['RMSE']:.4f}  NLL={d['NLL']:.4f}  viol={d['viol_prob']:.2e}")

    softplus_results = [d for d in results if d["h"] == "softplus"]
    best_softplus = min(softplus_results, key=lambda d: d["NLL"])
    best_softplus_r = best_softplus["r"]
    print(f"\n[md ablation] using h=softplus and best rank from the scan: r={best_softplus_r}")

    # md_list = [2, 4, 6, 8, 10, 12, 16]
    md_list = [50, 60, 70, 80, 90, 100]
    md_results = run_md_ablation(
        X,
        y,
        X_test,
        noise_std=noise_std,
        md_list=md_list,
        sf2=10.0,
        ell=1.5,
        rank_r=best_softplus_r,
        h_name="softplus",
        steps=1500,
        lr=2e-2,
        mc_train=64,
        mc_pred=512,
        seed=123,
    )

    print("\n=== md Ablation (softplus) ===")
    print("md  | RMSE    | NLL     | TIME(s)")
    print("-" * 33)
    for d in md_results:
        print(f"{d['md']:>2d}  | {d['RMSE']:.4f} | {d['NLL']:.4f} | {d['train_sec']:.1f}")

    md_vals = [d["md"] for d in md_results]
    md_rmse = [d["RMSE"] for d in md_results]
    md_nll = [d["NLL"] for d in md_results]
    md_time = [d["train_sec"] for d in md_results]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(md_vals, md_rmse, marker="o", color="forestgreen", linewidth=2)
    axes[0].set_title("RMSE vs md")
    axes[0].set_xlabel("md")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(md_vals, md_nll, marker="o", color="crimson", linewidth=2)
    axes[1].set_title("NLL vs md")
    axes[1].set_xlabel("md")
    axes[1].set_ylabel("NLL")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(md_vals, md_time, marker="o", color="steelblue", linewidth=2)
    axes[2].set_title("Time vs md")
    axes[2].set_xlabel("md")
    axes[2].set_ylabel("train seconds")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Constraint-point ablation (softplus, r={best_softplus_r})")
    fig.tight_layout()
    plt.show()

    print(f"\n[mf ablation] using h=softplus, r={best_softplus_r}, md={md}")

    mf_list = [10, 15, 20, 30, 40, 50]
    mf_results = run_mf_ablation(
        mf_list=mf_list,
        X_test=X_test,
        noise_std=noise_std,
        md=md,
        sf2=10.0,
        ell=1.5,
        rank_r=best_softplus_r,
        h_name="softplus",
        steps=1500,
        lr=2e-2,
        mc_train=64,
        mc_pred=512,
        seed=0,
    )

    print("\n=== mf Ablation (softplus) ===")
    print("mf  | RMSE    | NLL     | TIME(s)")
    print("-" * 33)
    for d in mf_results:
        print(f"{d['mf']:>2d}  | {d['RMSE']:.4f} | {d['NLL']:.4f} | {d['train_sec']:.1f}")

    mf_vals = [d["mf"] for d in mf_results]
    mf_rmse = [d["RMSE"] for d in mf_results]
    mf_nll = [d["NLL"] for d in mf_results]
    mf_time = [d["train_sec"] for d in mf_results]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(mf_vals, mf_rmse, marker="o", color="darkorange", linewidth=2)
    axes[0].set_title("RMSE vs mf")
    axes[0].set_xlabel("mf")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(mf_vals, mf_nll, marker="o", color="purple", linewidth=2)
    axes[1].set_title("NLL vs mf")
    axes[1].set_xlabel("mf")
    axes[1].set_ylabel("NLL")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(mf_vals, mf_time, marker="o", color="teal", linewidth=2)
    axes[2].set_title("Time vs mf")
    axes[2].set_xlabel("mf")
    axes[2].set_ylabel("train seconds")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Training-point ablation (softplus, r={best_softplus_r}, md={md})")
    fig.tight_layout()
    plt.show()


    

if __name__ == "__main__":
    main()
