"""Fit the stick-breaking Pyro mixture on real data (pbmc3k) via Adam/SVI,
and score it on the paper's own objective. See _common.py for the shared
pieces (stick_breaking, dm_total_loglik, the two init modes) and
pbmc3k_cavi.py for the closed-form-CAVI alternative to the Adam loop here.

The per-cell log-likelihood is a single matmul (`x @ log_alpha.T`, shape
(N, K)) computed once *before* the "cells" plate, not the elementwise
broadcast-then-sum simulate_and_fit.py uses for its tiny synthetic G=50
case. Both are mathematically the same thing (Pyro's parallel enumeration
for a plain Categorical walks its support in order 0..K-1, so the "K" axis
of the matmul output already *is* the enumeration axis -- no gather by `z`
needed), but the elementwise version would materialise a (K, N, G) tensor
at G=13,714; the matmul is O(N*K) in memory, O(N*G*K) flops as one GEMM.

Theta co-adapts with the partition (a `pyro.param` point estimate,
optimized jointly by the same SVI step) instead of being pinned to the
baseline's fitted value -- phi stays fixed (the genome-wide profile,
data-derived, not something either method treats as free), mirroring how
exact/fast also just MLE-fit Theta (docs/changes.md's "Fitting Theta").
"""

import time

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from sklearn.metrics import adjusted_rand_score
from torch.distributions import constraints

from _common import (
    hard_labels_and_score,
    init_tau_alpha,
    load_data,
    save_fit,
    stick_breaking,
)


@config_enumerate
def model(x, phi, k, gamma, theta_init, init_mode=None, seed=0):
    # init_mode/seed: unused here, SVI.step passes the same args to model
    # and guide, and only the guide's tau_alpha init needs them.
    theta = pyro.param("theta", torch.tensor(float(theta_init)), constraint=constraints.positive)
    theta_vec = theta * phi  # (G,)
    with pyro.plate("sticks", k - 1):
        v = pyro.sample("v", dist.Beta(1.0, gamma))
    pi = stick_breaking(v)
    with pyro.plate("states", k):
        alpha = pyro.sample("alpha", dist.Dirichlet(theta_vec))
    log_alpha = torch.log(alpha)  # (k, G)
    loglik_table = x @ log_alpha.T  # (N, k), computed once -- not per enum branch
    with pyro.plate("cells", x.shape[0]):
        pyro.sample("z", dist.Categorical(pi))
        pyro.factor("x", loglik_table.T)  # (k, N): already aligned with the enum axis


def guide(x, phi, k, gamma, theta_init, init_mode, seed):
    tau_a = pyro.param("tau_a", torch.ones(k - 1), constraint=constraints.positive)
    tau_b = pyro.param("tau_b", torch.full((k - 1,), float(gamma)), constraint=constraints.positive)
    with pyro.plate("sticks", k - 1):
        pyro.sample("v", dist.Beta(tau_a, tau_b))
    # lambda: pyro.param only evaluates the init value once (first call);
    # eagerly computing the (k, G) gather every step would be wasteful.
    tau_alpha = pyro.param(
        "tau_alpha", lambda: init_tau_alpha(x, phi, theta_init, k, init_mode, seed),
        constraint=constraints.positive,
    )
    with pyro.plate("states", k):
        pyro.sample("alpha", dist.Dirichlet(tau_alpha))


def fit(x, phi, k, gamma, theta_init, n_steps, init_mode="flat", lr=0.05, seed=0):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=TraceEnum_ELBO(max_plate_nesting=1))
    t_step = time.perf_counter()
    for step in range(n_steps):
        loss = svi.step(x, phi, k, gamma, theta_init, init_mode, seed)
        if step % 1000 == 0:
            dt = time.perf_counter() - t_step
            theta_now = pyro.param("theta").item()
            print(f"  step {step:5d}  elbo loss {loss:.1f}  theta {theta_now:.1f}"
                  f"  ({dt / max(step, 1) * 100:.2f}s/100steps)")


def posterior_labels(x, k):
    tau_a, tau_b = pyro.param("tau_a").detach(), pyro.param("tau_b").detach()
    pi_mean = stick_breaking(tau_a / (tau_a + tau_b))
    tau_alpha = pyro.param("tau_alpha").detach()
    alpha_mean = tau_alpha / tau_alpha.sum(-1, keepdim=True)  # (k, G)
    log_joint = torch.log(pi_mean).unsqueeze(0) + x @ torch.log(alpha_mean).T  # (N, k)
    return log_joint.argmax(-1).cpu().numpy()


def main(init_mode="real_cell", k_fit=None, n_steps=10_000, out="experiments/pyro_mixture/pyro_fit_B.npz"):
    counts_gN, x_t, phi_t, theta_scalar, lam, base = load_data()
    n = counts_gN.shape[1]
    k_fit = k_fit or n  # literal singleton: one state per cell when init_mode="real_cell"

    print(f"device: {x_t.device}")
    print(f"N={n} cells, G={counts_gN.shape[0]} genes, K_fit={k_fit}, init={init_mode}")
    print(f"baseline: {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}, "
          f"{float(base['seconds']):.1f}s\n")

    t0 = time.perf_counter()
    fit(x_t, phi_t, k_fit, gamma=5.0, theta_init=theta_scalar, n_steps=n_steps, init_mode=init_mode)
    dt = time.perf_counter() - t0

    fitted_theta = pyro.param("theta").item()
    labels = posterior_labels(x_t, k_fit)
    n_live, pyro_ll = hard_labels_and_score(counts_gN, labels, fitted_theta, lam)
    ari = adjusted_rand_score(base["labels"], labels)

    print(f"\npyro ({init_mode}, Adam): {n_live} states, LL={pyro_ll:.1f}, {dt:.1f}s, "
          f"theta {theta_scalar:.1f} -> {fitted_theta:.1f}")
    print(f"baseline (fast, res=1.0): {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}, "
          f"{float(base['seconds']):.1f}s")
    print(f"LL gap (pyro - baseline): {pyro_ll - float(base['log_likelihood']):.1f}")
    print(f"ARI vs baseline labels: {ari:.3f}")

    save_fit(out, labels, fitted_theta, pyro_ll, n_live, dt)


if __name__ == "__main__":
    main()
