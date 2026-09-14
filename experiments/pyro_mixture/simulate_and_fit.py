"""Simplest possible check for docs/ideas/pyro_dp_mixture.md: does enumeration
+ SVI recover known clusters under the model

    pi ~ Dirichlet(eta), alpha_s ~ Dirichlet(theta),
    z_c ~ Categorical(pi), x_c | z_c=s ~ Multinomial(N_c, alpha_s)

Finite K, flat Dirichlet(eta) on pi -- no stick-breaking prior yet, no real
data, no wiring into fastcellstates. Data is simulated from this exact model,
so recovering it is a lower bar than anything on real counts.

The Multinomial likelihood is written by hand (pyro.factor) rather than via
dist.Multinomial: torch's Multinomial needs one shared total_count for the
whole batch, but N_c varies per cell here as it would on real data. The
multinomial coefficient (constant in the latents, for fixed x_c) is dropped;
it cancels in both the ELBO's gradient and the responsibilities.
"""

import time

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from sklearn.metrics import adjusted_rand_score
from torch.distributions import constraints

torch.set_default_dtype(torch.float64)


def simulate(K_true=4, G=50, N=500, theta_scale=20.0, seed=0):
    rng = np.random.default_rng(seed)
    pi_true = rng.dirichlet(np.full(K_true, 5.0))
    phi = rng.dirichlet(np.full(G, 1.0))
    theta = theta_scale * phi
    alpha_true = rng.dirichlet(theta, size=K_true)  # (K_true, G)
    z_true = rng.choice(K_true, size=N, p=pi_true)
    n_c = rng.poisson(2000, size=N) + 200
    x = np.stack([rng.multinomial(n_c[c], alpha_true[z_true[c]]) for c in range(N)])
    return x.astype(np.int64), n_c.astype(np.int64), z_true, theta


@config_enumerate
def model(x, theta, k):
    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(k)))
    with pyro.plate("states", k):
        alpha = pyro.sample("alpha", dist.Dirichlet(theta))
    log_alpha = torch.log(alpha)  # (k, G)
    with pyro.plate("cells", x.shape[0]):
        z = pyro.sample("z", dist.Categorical(pi))
        loglik = (x * log_alpha[z]).sum(-1)  # broadcasts enumeration dim; drops the -> (k, N)
        pyro.factor("x", loglik)  # multinomial coefficient (constant in latents)


def guide(x, theta, k):
    tau_pi = pyro.param("tau_pi", torch.ones(k), constraint=constraints.positive)
    pyro.sample("pi", dist.Dirichlet(tau_pi))
    tau_alpha = pyro.param(
        "tau_alpha", theta.expand(k, -1).clone(), constraint=constraints.positive
    )
    with pyro.plate("states", k):
        pyro.sample("alpha", dist.Dirichlet(tau_alpha))


def fit(x, theta, k, n_steps=2000, lr=0.05, seed=0):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    x_t = torch.as_tensor(x)
    theta_t = torch.as_tensor(theta)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=TraceEnum_ELBO(max_plate_nesting=1))
    losses = []
    for step in range(n_steps):
        losses.append(svi.step(x_t, theta_t, k))
        if step % 500 == 0:
            print(f"  step {step:5d}  elbo loss {losses[-1]:.1f}")
    return losses


def posterior_labels(x, theta, k):
    x_t = torch.as_tensor(x)
    tau_pi = pyro.param("tau_pi").detach()
    tau_alpha = pyro.param("tau_alpha").detach()
    pi_mean = tau_pi / tau_pi.sum()
    alpha_mean = tau_alpha / tau_alpha.sum(-1, keepdim=True)  # (k, G)
    log_joint = torch.log(pi_mean).unsqueeze(1) + (
        x_t.unsqueeze(0) * torch.log(alpha_mean).unsqueeze(1)
    ).sum(-1)  # (k, N)
    labels = log_joint.argmax(0).numpy()
    return labels, pi_mean.numpy()


def main():
    k_true, g, n = 4, 50, 500
    k_fit = 8  # deliberately over-truncated
    x, n_c, z_true, theta = simulate(K_true=k_true, G=g, N=n)

    print(f"simulated N={n} cells, G={g} genes, K_true={k_true}; fitting K={k_fit}")
    t0 = time.perf_counter()
    fit(x, theta, k_fit)
    dt = time.perf_counter() - t0

    labels, pi_mean = posterior_labels(x, theta, k_fit)
    ari = adjusted_rand_score(z_true, labels)
    live = np.sum(pi_mean > 1.0 / (2 * k_fit))

    print(f"\nfit in {dt:.1f}s")
    print(f"posterior mixing weights: {np.round(pi_mean, 3)}")
    print(f"live states (weight > 1/2K): {live} / {k_fit} (K_true={k_true})")
    print(f"ARI vs true labels: {ari:.3f}")


if __name__ == "__main__":
    main()
