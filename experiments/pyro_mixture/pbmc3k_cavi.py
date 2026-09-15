"""Closed-form CAVI for the same stick-breaking mixture, instead of
Adam/SVI (pbmc3k_fit.py). The model is fully conjugate: z is already
handled exactly by enumeration in *both* versions (that doesn't change
here) -- what changes is alpha_s and the stick weights v_k, which both have
closed-form coordinate updates given the responsibilities (Blei & Jordan
2006's DP-mixture VI), instead of an Adam gradient step on their
variational parameters. No Pyro machinery needed: this is matmul + softmax
+ cumsum in a loop, each round guaranteed not to decrease the ELBO (unlike
Adam, which can oscillate).

One correctness detail that's easy to get wrong: the E-step responsibility
must use E_q[log alpha] (digamma-based), not log(mean alpha) -- the latter
is what pbmc3k_fit.py's posterior_labels uses as a cheap plug-in
approximation after training, fine there, but wrong as the actual CAVI
E-step.

Theta: two interchangeable coordinate updates for the same closed-form
objective J(theta) (the Dirichlet-prior/q(alpha) cross-entropy term, the
only place theta appears), both given the freshly-updated alpha's E[log
alpha] each round --
  "gradient": a few Adam steps on log(theta) each round.
  "linesearch": a Brent line search on log(theta) each round (scipy, same
    approach fastcellstates.moves.log_search uses to fit Theta for the
    exact/fast presets, just applied per-CAVI-round instead of per-recluster).
"""

import time

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import gammaln as np_gammaln
from sklearn.metrics import adjusted_rand_score

from _common import hard_labels_and_score, init_tau_alpha, load_data, save_fit, stick_breaking


def _elog_pi(ta, tb, k, device):
    elog_v = torch.digamma(ta) - torch.digamma(ta + tb)
    elog_1mv = torch.digamma(tb) - torch.digamma(ta + tb)
    cum_incl = torch.cumsum(elog_1mv, dim=0)
    elog_pi = torch.empty(k, device=device, dtype=ta.dtype)
    elog_pi[: k - 1] = elog_v + (cum_incl - elog_1mv)
    elog_pi[k - 1] = cum_incl[-1]
    return elog_pi


def _theta_neg_obj(log_theta, phi_np, s_np, k):
    theta = np.exp(log_theta)
    return -(k * np_gammaln(theta) - k * np_gammaln(theta * phi_np).sum() + theta * (phi_np * s_np).sum())


def _theta_step_linesearch(theta, phi_np, s_np, k):
    log_theta0 = float(np.log(theta))
    res = minimize_scalar(
        _theta_neg_obj, args=(phi_np, s_np, k), method="brent",
        bracket=(log_theta0 - 1, log_theta0, log_theta0 + 1),
    )
    return float(np.exp(res.x))


def _theta_step_gradient(theta, phi, s, k, n_inner=20, lr=0.1):
    log_theta = torch.log(torch.as_tensor(theta, device=phi.device)).clone().requires_grad_(True)
    opt = torch.optim.Adam([log_theta], lr=lr)
    for _ in range(n_inner):
        opt.zero_grad()
        th = torch.exp(log_theta)
        j = k * torch.lgamma(th) - k * torch.lgamma(th * phi).sum() + th * (phi * s).sum()
        (-j).backward()
        opt.step()
    return float(torch.exp(log_theta).detach())


def fit_cavi(x, phi, k, gamma, theta_init, n_rounds, init_mode, theta_mode, seed=0, print_every=20):
    device = x.device
    tau_alpha = init_tau_alpha(x, phi, theta_init, k, init_mode, seed)
    ta = torch.ones(k - 1, device=device)
    tb = torch.full((k - 1,), float(gamma), device=device)
    theta = float(theta_init)
    phi_np = phi.cpu().numpy()

    for r in range(n_rounds):
        # E-step: responsibilities from E_q[log alpha], E_q[log pi] (digamma-based, not log-of-mean)
        elog_alpha = torch.digamma(tau_alpha) - torch.digamma(tau_alpha.sum(-1, keepdim=True))
        elog_pi = _elog_pi(ta, tb, k, device)
        logr = elog_pi.unsqueeze(0) + x @ elog_alpha.T  # (N, k)
        resp = torch.softmax(logr, dim=-1)

        # M-step alpha: closed form (Dirichlet-multinomial conjugacy)
        tau_alpha = theta * phi.unsqueeze(0) + resp.T @ x  # (k, G)

        # M-step v: closed form (Beta-stick conjugacy, Blei & Jordan 2006)
        n_k = resp.sum(0)  # (k,)
        cum_incl_n = torch.cumsum(n_k, dim=0)
        tail = cum_incl_n[-1] - cum_incl_n[:-1]
        ta = 1.0 + n_k[:-1]
        tb = gamma + tail

        # M-step theta: uses the just-updated alpha's E[log alpha]
        elog_alpha_new = torch.digamma(tau_alpha) - torch.digamma(tau_alpha.sum(-1, keepdim=True))
        s = elog_alpha_new.sum(0)  # (G,)
        if theta_mode == "gradient":
            theta = _theta_step_gradient(theta, phi, s, k)
        elif theta_mode == "linesearch":
            theta = _theta_step_linesearch(theta, phi_np, s.cpu().numpy(), k)
        else:
            raise ValueError(theta_mode)

        if r % print_every == 0:
            labels = logr.argmax(-1).cpu().numpy()
            print(f"  round {r:4d}  theta {theta:.1f}  live states {len(np.unique(labels))}")

    return tau_alpha, ta, tb, theta


def posterior_labels(x, tau_alpha, ta, tb):
    pi_mean = stick_breaking(ta / (ta + tb))
    alpha_mean = tau_alpha / tau_alpha.sum(-1, keepdim=True)
    log_joint = torch.log(pi_mean).unsqueeze(0) + x @ torch.log(alpha_mean).T
    return log_joint.argmax(-1).cpu().numpy()


def run(init_mode, theta_mode, k_fit=None, n_rounds=200, out=None):
    counts_gN, x_t, phi_t, theta_scalar, lam, base = load_data()
    n = counts_gN.shape[1]
    k_fit = k_fit or n

    print(f"device: {x_t.device}")
    print(f"N={n} cells, G={counts_gN.shape[0]} genes, K_fit={k_fit}, "
          f"init={init_mode}, theta={theta_mode}")
    print(f"baseline: {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}, "
          f"{float(base['seconds']):.1f}s\n")

    t0 = time.perf_counter()
    tau_alpha, ta, tb, fitted_theta = fit_cavi(
        x_t, phi_t, k_fit, gamma=5.0, theta_init=theta_scalar,
        n_rounds=n_rounds, init_mode=init_mode, theta_mode=theta_mode,
    )
    dt = time.perf_counter() - t0

    labels = posterior_labels(x_t, tau_alpha, ta, tb)
    n_live, pyro_ll = hard_labels_and_score(counts_gN, labels, fitted_theta, lam)
    ari = adjusted_rand_score(base["labels"], labels)

    print(f"\npyro ({init_mode}, CAVI-{theta_mode}): {n_live} states, LL={pyro_ll:.1f}, {dt:.1f}s, "
          f"theta {theta_scalar:.1f} -> {fitted_theta:.1f}")
    print(f"baseline (fast, res=1.0): {int(base['n_states'])} states, LL={float(base['log_likelihood']):.1f}")
    print(f"LL gap (pyro - baseline): {pyro_ll - float(base['log_likelihood']):.1f}")
    print(f"ARI vs baseline labels: {ari:.3f}")

    out = out or f"experiments/pyro_mixture/pyro_fit_cavi_{init_mode}_{theta_mode}.npz"
    save_fit(out, labels, fitted_theta, pyro_ll, n_live, dt)


if __name__ == "__main__":
    # C: flat init, CAVI -- isolates the optimizer effect alone
    run(init_mode="flat", theta_mode="gradient",
        out="experiments/pyro_mixture/pyro_fit_C.npz")
