# A Pyro-fit truncated mixture, as an alternative to the partition search

Status: explored, not adopted. Originally written as a scratch note in the
gitignored `docs/ideas/`; moved here once it had actually been tested, so
the motivating idea and the results that came out of it live in the same
place. See `README.md` in this directory for the full experimental log and
the conclusion.

## Why

`fastcellstates`/`cellstates` finds a partition $\rho$ by combinatorial search
(MCMC from singletons, or Leiden warm start + merge + sweep) over the exact
Dirichlet-multinomial marginal likelihood (`docs/changes.md`, eq. 15). That
search is exact in the sense that every accepted move strictly improves a
well-defined objective, but it is inherently sequential per move and doesn't
subsample cells.

An alternative: cast the same generative model as a Bayesian mixture and fit
it with stochastic variational inference (SVI) in Pyro/PyTorch. This trades
the exact combinatorial objective for a differentiable, minibatchable one —
a genuinely different algorithm, not a speedup of the existing search. The
appeal is largely orthogonal to the current speedups: SVI scales to $N$ via
minibatches regardless of $K$, which the current merge/sweep machinery does
not.

## Generative model

Fix the Dirichlet prior over gene frequencies exactly as the base model does:
$\theta = \Theta\phi$, with $\phi_g$ the genome-wide UMI fractions and
$\Theta$ the one free scale. Truncate the mixture at $K$ states (an upper
bound, not a target count).

**Finite-$K$ version**, matching the formulation in the prior discussion:

$$
\pi \sim \mathrm{Dirichlet}(\eta\mathbf{1}_K), \qquad
\alpha_s \sim \mathrm{Dirichlet}(\theta), \quad s = 1,\dots,K
$$

$$
z_c \sim \mathrm{Categorical}(\pi), \qquad
x_c \mid z_c{=}s,\alpha_s \sim \mathrm{Multinomial}(N_c,\alpha_s), \quad c = 1,\dots,N
$$

where $N_c = \sum_g x_{gc}$ is the observed per-cell total (conditioned on,
exactly as the base model conditions on $N_s$).

**Truncated stick-breaking**, to keep the "don't have to guess $K$" property
that makes the base model attractive, replacing the flat Dirichlet on $\pi$:

$$
v_k \sim \mathrm{Beta}(1,\gamma), \ k=1,\dots,K{-}1, \qquad v_K := 1, \qquad
\pi_k = v_k\!\!\prod_{j<k}(1-v_j)
$$

$K$ only needs to be generously larger than the expected number of occupied
states; unused states get $\pi_k \to 0$ in the posterior. $\gamma$
(concentration) is the DP's usual knob on the expected number of occupied
states; it can be fixed or given its own $\mathrm{Gamma}(a_0,b_0)$ prior (see
[Open questions](#open-questions)).

Joint density (finite-$K$ form, $\pi$ from either construction):

$$
p(x,z,\alpha,\pi) = \Big[\textstyle\prod_{s=1}^K \mathrm{Dir}(\alpha_s;\theta)\Big]\cdot p(\pi)\cdot
\prod_{c=1}^N \pi_{z_c}\,\mathrm{Mult}(x_c;N_c,\alpha_{z_c})
$$

### Relation to the exact model's marginal

For a *fixed, hard* partition $\rho$, integrating one shared $\alpha_s$
jointly over every cell of subset $s$ collapses exactly to the closed form
the exact search uses (eq. 15), because the Multinomial likelihood only
depends on $\alpha_s$ through $\prod_g \alpha_{gs}^{n_{gs}}$ — the aggregated
counts $n_{gs}=\sum_{c\in s}x_{gc}$ are a sufficient statistic. That collapse
is what makes the exact search's incremental merge/sweep deltas $O(1)$ in the
number of cells per move.

Here $z_c$ is not fixed — it's marginalized simultaneously over *every* cell
via enumeration — so there is no single hard membership to aggregate against
before integrating $\alpha_s$ out. Mean-field VI instead keeps $\alpha_s$ as
an explicit latent with its own variational posterior. This is a real
structural difference, not just an implementation choice: it's why this is a
different model-fitting procedure, not a re-derivation of the same objective.

## Inference

$z_c$ is a $K$-way categorical — Pyro can marginalize it **exactly** via
enumeration (`pyro.infer.config_enumerate` + `TraceEnum_ELBO`), no
Gumbel-softmax relaxation needed, the same pattern as Pyro's own GMM
tutorial. The continuous latents get a conjugate mean-field guide:

$$
q(v_k) = \mathrm{Beta}(a_k,b_k), \qquad q(\alpha_s) = \mathrm{Dirichlet}(\tau_s)
$$

ELBO, with the $z_c$ sum done exactly (not sampled):

$$
\mathcal L = \mathbb E_{q(v,\alpha)}\Big[\sum_{c=1}^N \log\!\sum_{s=1}^K \pi_s(v)\,\mathrm{Mult}(x_c;N_c,\alpha_s)
+ \log p(v) + \log p(\alpha) - \log q(v) - \log q(\alpha)\Big]
$$

Both $q(v_k)$ and $q(\alpha_s)$ are reparameterizable (`torch.distributions`
supports `rsample` for Beta and Dirichlet), so this is a standard
enumeration + reparameterized-gradient SVI setup, minibatchable over cells
via `pyro.plate("cells", N, subsample_size=B)`.

## What it would need

A standalone prototype (Pyro model is a per-cell generative program, not a
mutate-in-place partition — it doesn't fit the `Cluster`/`moves` machinery,
so this would *not* plug into the existing merge/sweep search):

- `phi_and_theta(counts, theta) -> (phi, theta_vec)` — reuse
  `model._dm_kernels`'s existing prior construction, so $\phi_g$ matches the
  exact path and results are comparable.
- `pyro_model(counts, N_c, K, theta_vec, eta_or_gamma)` — the generative
  program above: `pyro.plate("states", K)` for $v_k,\alpha_s$; enumerated
  `pyro.plate("cells", N, subsample_size=B)` for $z_c,x_c$; `Beta`,
  `Dirichlet`, `Categorical`, `Multinomial`.
- `pyro_guide(...)` — mean-field guide: `pyro.param` for $(a_k,b_k)$ and
  $\tau_s$ (positivity-constrained), sampling `Beta`/`Dirichlet` from them.
  ($z_c$ needs no guide term — enumeration marginalizes it directly.)
- `fit(counts, K, theta, gamma, n_steps, batch_size, lr) -> FitResult` —
  `SVI(model, guide, Adam(lr), TraceEnum_ELBO(max_plate_nesting=1))`, looped
  `svi.step` over minibatches.
- `posterior_responsibilities(fit, counts) -> (N, K) array` — $q(z_c{=}s)
  \propto \pi_s\,\mathrm{Mult}(x_c;N_c,\alpha_s)$, for hard labels and for a
  `predict_state`-style per-cell scoring path.
- `to_summary(fit, genes, cells, min_weight) -> Summary` — drop states with
  $\mathbb E_q[\pi_s] <$ `min_weight`; the fitted $\tau_s$ *is* already a
  posterior Dirichlet parameter, the same quantity the existing `Summary`
  stores as $\Theta\phi_g+n_{gs}$ (eq. 18) — so a fitted mixture could
  populate a real `Summary` and get `.sample()`/`.reconstruct()`/
  `.predict_state()` for free, without touching those methods.

New dependency: `pyro-ppl` (pulls in `torch`) — a large, heavy addition
right after the floors work tightened the dependency story elsewhere in this
project. Should stay an optional extra (e.g. `fastcellstates[pyro]`) if this
ever leaves the prototype stage; GPU is a nice-to-have, not a requirement,
since SVI minibatches regardless.

## Evaluation plan

Against `exact` and `fast` on the same benchmark datasets
(`benchmarks/bench_vs_upstream`): occupied-state count, ARI/AMI of hard
labels, wall-clock and peak memory as a function of $N$, and sensitivity to
$K$ (truncation) and $\gamma$ (concentration).

## Open questions

- **$\gamma$ is a real dial.** A DP-style prior still needs a concentration
  parameter controlling the expected number of occupied states
  ($\sim\gamma\log N$); a weak hyperprior $\mathrm{Gamma}(a_0,b_0)$ softens
  this but doesn't remove it. This is a direct tension with the "no dials to
  turn" property the base model is valued for (`docs/changes.md`).
- **$K$ needs to be a generous fixed ceiling**, and there's no analogue of
  the original's from-singletons growth — a dataset whose true state count
  exceeds $K$ will silently saturate the truncation rather than fail loudly.
- **The ELBO is a lower bound**, not the exact objective the merge search
  climbs; no guarantee of matching or beating its optimum, and mean-field VI
  is sensitive to initialization and minibatch noise in ways the closed-form
  greedy search isn't.
- **Sparsity.** The whole first section of `docs/changes.md` is about
  restricting every update to a cell's *nonzero* genes. A dense per-cell,
  per-state $\log\mathrm{Mult}(x_c;N_c,\alpha_s)$ costs $O(K\cdot G)$ unless
  the Multinomial log-prob is written to gather over each cell's nonzero
  indices — `torch`'s sparse-tensor autograd support for this pattern is
  much less mature than dense ops, and reproducing the sparsity win here is
  not automatic.
