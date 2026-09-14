"""Simplest possible check for docs/ideas/pyro_dp_mixture.md: does enumeration
+ SVI recover known clusters under the model

    pi ~ Dirichlet(eta) or truncated stick-breaking, alpha_s ~ Dirichlet(theta),
    z_c ~ Categorical(pi), x_c | z_c=s ~ Multinomial(N_c, alpha_s)

`gamma=None` uses a flat Dirichlet(1) on pi (the first pass); a `gamma` value
switches to the truncated stick-breaking prior from the ideas note:
v_k ~ Beta(1, gamma), pi_k = v_k * prod_{j<k}(1-v_j). gamma is left as a
fixed knob here, not itself given a hyperprior -- exactly the "still a real
dial" point the ideas note flags. No real data, no wiring into
fastcellstates. Data is simulated from the (Dirichlet-pi) model, so
recovering it is a lower bar than anything on real counts.

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)  # tensors pyro/torch create internally (Beta(1.0, gamma) etc.) follow this too


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


def stick_breaking(v):
    """(..., K-1) sticks in (0,1) -> (..., K) simplex weights.

    pi_k = v_k * prod_{j<k}(1-v_j) for k<K, pi_K = prod_{j<K}(1-v_j) (v_K:=1
    implicit -- the last piece is whatever remains of the stick).
    """
    remaining = torch.cumprod(1 - v, dim=-1)
    pi_head = torch.cat([v[..., :1], v[..., 1:] * remaining[..., :-1]], dim=-1)
    return torch.cat([pi_head, remaining[..., -1:]], dim=-1)


@config_enumerate
def model(x, theta, k, gamma=None):
    if gamma is None:
        pi = pyro.sample("pi", dist.Dirichlet(torch.ones(k)))
    else:
        with pyro.plate("sticks", k - 1):
            v = pyro.sample("v", dist.Beta(1.0, gamma))
        pi = stick_breaking(v)
    with pyro.plate("states", k):
        alpha = pyro.sample("alpha", dist.Dirichlet(theta))
    log_alpha = torch.log(alpha)  # (k, G)
    with pyro.plate("cells", x.shape[0]):
        z = pyro.sample("z", dist.Categorical(pi))
        loglik = (x * log_alpha[z]).sum(-1)  # broadcasts enumeration dim -> (k, N)
        pyro.factor("x", loglik)  # multinomial coefficient (constant in latents)


def guide(x, theta, k, gamma=None):
    if gamma is None:
        tau_pi = pyro.param("tau_pi", torch.ones(k), constraint=constraints.positive)
        pyro.sample("pi", dist.Dirichlet(tau_pi))
    else:
        tau_a = pyro.param("tau_a", torch.ones(k - 1), constraint=constraints.positive)
        tau_b = pyro.param(
            "tau_b", torch.full((k - 1,), float(gamma)), constraint=constraints.positive
        )
        with pyro.plate("sticks", k - 1):
            pyro.sample("v", dist.Beta(tau_a, tau_b))
    tau_alpha = pyro.param(
        "tau_alpha", theta.expand(k, -1).clone(), constraint=constraints.positive
    )
    with pyro.plate("states", k):
        pyro.sample("alpha", dist.Dirichlet(tau_alpha))


def fit(x, theta, k, gamma=None, n_steps=2000, lr=0.05, seed=0, name=""):
    pyro.clear_param_store()
    pyro.set_rng_seed(seed)
    x_t = torch.as_tensor(x, device=DEVICE)
    theta_t = torch.as_tensor(theta, device=DEVICE)
    svi = SVI(model, guide, Adam({"lr": lr}), loss=TraceEnum_ELBO(max_plate_nesting=1))
    losses = []
    for step in range(n_steps):
        losses.append(svi.step(x_t, theta_t, k, gamma))
        if step % 500 == 0:
            print(f"  [{name}] step {step:5d}  elbo loss {losses[-1]:.1f}")
    return losses


def posterior_pi(_k, gamma):
    if gamma is None:
        tau_pi = pyro.param("tau_pi").detach()
        return tau_pi / tau_pi.sum()
    tau_a = pyro.param("tau_a").detach()
    tau_b = pyro.param("tau_b").detach()
    v_mean = tau_a / (tau_a + tau_b)  # Beta posterior mean, plugged into the transform
    return stick_breaking(v_mean)


def posterior_labels(x, k, gamma):
    x_t = torch.as_tensor(x, device=DEVICE)
    pi_mean = posterior_pi(k, gamma)
    tau_alpha = pyro.param("tau_alpha").detach()
    alpha_mean = tau_alpha / tau_alpha.sum(-1, keepdim=True)  # (k, G)
    log_joint = torch.log(pi_mean).unsqueeze(1) + (
        x_t.unsqueeze(0) * torch.log(alpha_mean).unsqueeze(1)
    ).sum(-1)  # (k, N)
    labels = log_joint.argmax(0).cpu().numpy()
    return labels, pi_mean.cpu().numpy()


def run(name, x, theta, k, z_true, gamma=None, **fit_kwargs):
    t0 = time.perf_counter()
    fit(x, theta, k, gamma=gamma, name=name, **fit_kwargs)
    dt = time.perf_counter() - t0
    labels, pi_mean = posterior_labels(x, k, gamma)
    ari = adjusted_rand_score(z_true, labels)
    live = int(np.sum(pi_mean > 1.0 / (2 * k)))
    print(f"\n[{name}] fit in {dt:.1f}s")
    print(f"[{name}] posterior mixing weights: {np.round(pi_mean, 3)}")
    print(f"[{name}] live states (weight > 1/2K): {live} / {k}")
    print(f"[{name}] ARI vs true labels: {ari:.3f}\n")
    return {"name": name, "dt": dt, "pi_mean": pi_mean, "live": live, "ari": ari}


def main():
    k_true, g, n = 4, 50, 500
    k_fit = 8  # deliberately over-truncated
    x, _n_c, z_true, theta = simulate(K_true=k_true, G=g, N=n)
    print(f"device: {DEVICE}")
    print(f"simulated N={n} cells, G={g} genes, K_true={k_true}; fitting K={k_fit}\n")

    dirichlet = run("dirichlet-pi", x, theta, k_fit, z_true, gamma=None)
    stick = run("stick-breaking (gamma=1.0)", x, theta, k_fit, z_true, gamma=1.0)

    print(f"summary (K_true={k_true}):")
    for r in (dirichlet, stick):
        print(f"  {r['name']:<28} live={r['live']}  ari={r['ari']:.3f}  {r['dt']:.1f}s")


if __name__ == "__main__":
    main()
