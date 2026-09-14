"""Fit the stick-breaking Pyro mixture on real data (pbmc3k) and score it on
the paper's own objective -- not a synthetic sanity check anymore.

Two things changed from simulate_and_fit.py, both necessary at this scale
(G=13,714 genes, baseline found 635 states out of 2,700 cells):

1. The per-cell log-likelihood is now a single matmul (`x @ log_alpha.T`,
   shape (N, K)) computed once *before* the "cells" plate, instead of the
   elementwise-broadcast-then-sum used for the tiny synthetic G=50 case.
   Both are mathematically the same thing (Pyro's parallel enumeration for
   a plain Categorical walks its support in order 0..K-1, so the "K" axis
   of the matmul output already *is* the enumeration axis -- no gather by
   `z` needed), but the elementwise version would materialise a (K, N, G)
   tensor here; the matmul is O(N*K) in memory, O(N*G*K) flops as a single
   GEMM instead of an elementwise product. This is the "sparse-aware
   rewrite" flagged as an open question in docs/ideas/pyro_dp_mixture.md --
   not sparse yet (x is still a dense tensor), but the shape of the fix.

2. The score reported isn't ARI against a ground truth we don't have on
   real data -- it's the paper's own closed-form marginal log-likelihood
   (eq. 15), evaluated at the *baseline's own fitted Theta, phi* on the
   partition this script finds. `dm_total_loglik` below is a direct copy
   of `fastcellstates.model.DirichletMultinomial.cluster_loglik`'s formula
   (not imported: keeps this venv free of numba/fastcellstates as a
   dependency). Comparable number, same units, same data, same Theta as
   `baseline.npz` (see baseline.py, run separately under pixi).
"""

import time

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from scipy.special import gammaln
from sklearn.metrics import adjusted_rand_score
from torch.distributions import constraints

DATA = "experiments/pyro_mixture/pbmc3k_counts.npy"
BASELINE = "experiments/pyro_mixture/baseline.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)


def stick_breaking(v):
    remaining = torch.cumprod(1 - v, dim=-1)
    pi_head = torch.cat([v[..., :1], v[..., 1:] * remaining[..., :-1]], dim=-1)
    return torch.cat([pi_head, remaining[..., -1:]], dim=-1)


@config_enumerate
def model(x, theta, k, gamma):
    with pyro.plate("sticks", k - 1):
        v = pyro.sample("v", dist.Beta(1.0, gamma))
    pi = stick_breaking(v)
    with pyro.plate("states", k):
        alpha = pyro.sample("alpha", dist.Dirichlet(theta))
    log_alpha = torch.log(alpha)  # (k, G)
    loglik_table = x @ log_alpha.T  # (N, k), computed once -- not per enum branch
    with pyro.plate("cells", x.shape[0]):
        pyro.sample("z", dist.Categorical(pi))
        pyro.factor("x", loglik_table.T)  # (k, N): already aligned with the enum axis


def guide(x, theta, k, gamma):
    tau_a = pyro.param("tau_a", torch.ones(k - 1), constraint=constraints.positive)
    tau_b = pyro.param("tau_b", torch.full((k - 1,), float(gamma)), constraint=constraints.positive)
    with pyro.plate("sticks", k - 1):
        pyro.sample("v", dist.Beta(tau_a, tau_b))
    tau_alpha = pyro.param(
        "tau_alpha", theta.expand(k, -1).clone(), constraint=constraints.positive
    )
    with pyro.plate("states", k):
        pyro.sample("alpha", dist.Dirichlet(tau_alpha))


def fit(x, theta, k, gamma, n_steps, lr=0.05, seed=0):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=TraceEnum_ELBO(max_plate_nesting=1))
    t_step = time.perf_counter()
    for step in range(n_steps):
        loss = svi.step(x, theta, k, gamma)
        if step % 1000 == 0:
            dt = time.perf_counter() - t_step
            print(f"  step {step:5d}  elbo loss {loss:.1f}  ({dt / max(step, 1) * 100:.2f}s/100steps)")


def posterior_labels(x, k):
    tau_a, tau_b = pyro.param("tau_a").detach(), pyro.param("tau_b").detach()
    pi_mean = stick_breaking(tau_a / (tau_a + tau_b))
    tau_alpha = pyro.param("tau_alpha").detach()
    alpha_mean = tau_alpha / tau_alpha.sum(-1, keepdim=True)  # (k, G)
    log_joint = torch.log(pi_mean).unsqueeze(0) + x @ torch.log(alpha_mean).T  # (N, k)
    return log_joint.argmax(-1).cpu().numpy()


def dm_total_loglik(theta, lam, counts_per_state):
    """Mirrors fastcellstates.model.DirichletMultinomial.cluster_loglik
    (docs/changes.md eq. 15), summed over states with >0 assigned cells."""
    a = theta * lam
    B = gammaln(theta) - gammaln(a).sum()
    total = 0.0
    for c in counts_per_state:
        nc = c.sum()
        if nc == 0:
            continue
        total += B - gammaln(nc + theta) + gammaln(a + c).sum()
    return float(total)


def main():
    counts_gN = np.load(DATA)  # (G, N) int, matches fastcellstates' own convention
    base = np.load(BASELINE)
    theta_scalar, lam = float(base["theta"]), base["lam"]
    n = counts_gN.shape[1]
    k_fit = 700  # baseline ("fast", resolution=1.0) found 635/2700 states live

    print(f"device: {DEVICE}")
    print(f"N={n} cells, G={counts_gN.shape[0]} genes, K_fit={k_fit}")
    print(f"baseline: {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}, "
          f"{float(base['seconds']):.1f}s\n")

    x_t = torch.as_tensor(counts_gN.T, device=DEVICE, dtype=torch.get_default_dtype())  # (N, G)
    theta_t = torch.as_tensor(theta_scalar * lam, device=DEVICE)  # (G,), = Theta*phi

    t0 = time.perf_counter()
    fit(x_t, theta_t, k_fit, gamma=5.0, n_steps=10_000)
    dt = time.perf_counter() - t0

    labels = posterior_labels(x_t, k_fit)
    n_live = len(np.unique(labels))

    agg = np.zeros((k_fit, counts_gN.shape[0]))
    np.add.at(agg, labels, counts_gN.T)
    pyro_ll = dm_total_loglik(theta_scalar, lam, agg)

    ari = adjusted_rand_score(base["labels"], labels)

    print(f"\npyro (stick-breaking): {n_live} states, LL={pyro_ll:.1f}, {dt:.1f}s")
    print(f"baseline (fast, res=1.0): {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}, "
          f"{float(base['seconds']):.1f}s")
    print(f"LL gap (pyro - baseline): {pyro_ll - float(base['log_likelihood']):.1f}")
    print(f"ARI vs baseline labels: {ari:.3f}")


if __name__ == "__main__":
    main()
