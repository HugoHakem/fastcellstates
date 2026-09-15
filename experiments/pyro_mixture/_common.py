"""Shared pieces between pbmc3k_fit.py (Adam/SVI) and pbmc3k_cavi.py
(closed-form CAVI): device setup, the stick-breaking transform, the paper's
closed-form DM marginal log-likelihood (eq. 15), and the two initialization
modes ("flat" vs "real_cell" -- the singleton-init analogue)."""

import numpy as np
import torch
from scipy.special import gammaln

DATA = "experiments/pyro_mixture/pbmc3k_counts.npy"
BASELINE = "experiments/pyro_mixture/baseline.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)


def stick_breaking(v):
    """(..., K-1) sticks in (0,1) -> (..., K) simplex weights."""
    remaining = torch.cumprod(1 - v, dim=-1)
    pi_head = torch.cat([v[..., :1], v[..., 1:] * remaining[..., :-1]], dim=-1)
    return torch.cat([pi_head, remaining[..., -1:]], dim=-1)


def dm_total_loglik(theta, lam, counts_per_state):
    """Mirrors fastcellstates.model.DirichletMultinomial.cluster_loglik
    (docs/changes.md eq. 15), summed over states with >0 assigned cells."""
    a = theta * lam
    b = gammaln(theta) - gammaln(a).sum()
    total = 0.0
    for c in counts_per_state:
        nc = c.sum()
        if nc == 0:
            continue
        total += b - gammaln(nc + theta) + gammaln(a + c).sum()
    return float(total)


def init_tau_alpha(x, phi, theta_init, k, mode, seed=0):
    """Initial (k, G) Dirichlet variational parameter for alpha.

    "flat": every state starts at the prior mean theta*phi (fully symmetric
    -- SVI/CAVI has to break the symmetry from noise alone).

    "real_cell": each state seeded from one real cell's own counts,
    theta*phi + counts[cell_s] -- the singleton-init analogue: every
    initial state is anchored to one actual cell, not an artificial
    average. k == N gives literal singletons (a 1:1 bijection, no
    sampling); k < N samples k cells without replacement.
    """
    prior = theta_init * phi  # (G,)
    if mode == "flat":
        return prior.expand(k, -1).clone()
    if mode == "real_cell":
        n = x.shape[0]
        if k == n:
            idx = torch.arange(n, device=x.device)
        else:
            g = torch.Generator(device=x.device).manual_seed(seed)
            idx = torch.randperm(n, device=x.device, generator=g)[:k]
        return prior.unsqueeze(0) + x[idx]
    raise ValueError(mode)


def load_data():
    """(counts_gN, x_t (N,G) tensor, phi_t (G,) tensor, theta_scalar, lam, base npz)."""
    counts_gN = np.load(DATA)  # (G, N) int, matches fastcellstates' own convention
    base = np.load(BASELINE)
    theta_scalar, lam = float(base["theta"]), base["lam"]
    x_t = torch.as_tensor(counts_gN.T, device=DEVICE, dtype=torch.get_default_dtype())  # (N, G)
    phi_t = torch.as_tensor(lam, device=DEVICE)  # (G,), fixed
    return counts_gN, x_t, phi_t, theta_scalar, lam, base


def hard_labels_and_score(x_np_gN, labels, theta, lam):
    """(n_live, log_likelihood) for a hard label assignment, via the eq. 15 formula."""
    k = int(labels.max()) + 1
    agg = np.zeros((k, x_np_gN.shape[0]))
    np.add.at(agg, labels, x_np_gN.T)
    return len(np.unique(labels)), dm_total_loglik(theta, lam, agg)


def save_fit(path, labels, theta, log_likelihood, n_states, seconds):
    np.savez(path, labels=labels, theta=theta, log_likelihood=log_likelihood,
              n_states=n_states, seconds=seconds)
    print(f"wrote {path}")
