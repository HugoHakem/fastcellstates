# Pyro mixture experiment

Scratch prototype for the idea in [`docs/ideas/pyro_dp_mixture.md`](../../docs/ideas/pyro_dp_mixture.md):
fit the same generative model (Categorical/Dirichlet/Multinomial) with Pyro's
SVI instead of the exact partition search. Deliberately not wired into
`fastcellstates` — no shared code, no CLI, no `Summary` integration yet. Just
checking the inference procedure itself works before anything else.

Isolated on purpose: `pyro-ppl` (+ `torch`) is not a project dependency and
this doesn't use pixi's environment.

```sh
cd experiments/pyro_mixture
uv venv --python 3.11 .venv-pyro
uv pip install --python .venv-pyro/bin/python pyro-ppl scikit-learn
.venv-pyro/bin/python simulate_and_fit.py
```

`pip install pyro-ppl` pulls in whatever the currently-latest `torch` build is
(here: `+cu130`), which may not match this node's driver -- check
`nvidia-smi`'s "CUDA Version" and reinstall `torch` from the matching
PyTorch wheel index if `torch.cuda.is_available()` comes back `False`:

```sh
uv pip install --python .venv-pyro/bin/python --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
```

(`cu128` here because this node's driver reports CUDA 12.8; pick the index
matching whatever `nvidia-smi` reports elsewhere.) The script itself picks
up `cuda` automatically via `torch.cuda.is_available()` -- no separate flag.

## Status

`simulate_and_fit.py`: generates data from the exact model this is meant to
approximate (known `K_true`, `pi`, `alpha`, varying per-cell library size
`N_c`), fits a finite-`K` (over-truncated) mixture via enumeration + SVI
under either prior on `pi` — flat `Dirichlet(1)`, or the truncated
stick-breaking prior from the ideas note (`v_k ~ Beta(1, gamma)`, `gamma`
fixed, still the open dial the ideas note flags) — and reports ARI against
the true labels plus which states end up "live". Simplest possible check:
does the inference procedure itself recover known clusters at all, before
testing anything on real data.

First run, flat `Dirichlet(1)` on `pi` (`K_true=4`, `G=50`, `N=500`,
`K_fit=8`, 2000 SVI steps, default seed): ~8s on CPU, 3/4 true states end up
"live" (one absorbed into another), ARI 0.86.

Second run, same dataset, stick-breaking prior with `gamma=1.0`: all 4 true
states live, ARI 1.000, ~6s. One run each, so a data point rather than a
trend, but it's the direction the ideas note's reasoning predicted — a prior
biased toward few occupied components should make it easier for unused
truncation slots to actually collapse to ~0 weight instead of splitting mass
across them.

Next things worth actually varying before trusting this: seed/init
sensitivity for both variants (is the flat-Dirichlet merge, and the
stick-breaking clean recovery, each a fluke of one run or systematic?),
sensitivity to `gamma` and to `K_fit`, and only then real data.

GPU verified working (H200, `torch==2.11.0+cu128`): the script picks up
`cuda` automatically, ran end to end, matches a raw matmul smoke test. At
this synthetic problem's tiny size (N=500, G=50, K=8) it's actually slower
than CPU (~10s vs ~6s, kernel-launch overhead dominates) -- expected, and
irrelevant until the real-data / scaling step below, which is where a GPU
actually matters.

## Real data: pbmc3k

`baseline.py` (run under `pixi run python`, needs real `fastcellstates`) fits
the same `pbmc3k` counts (`experiments/pyro_mixture/pbmc3k_counts.npy`, via
`benchmarks/bench_vs_upstream/scripts/prepare_data.py`) with the `fast`
preset at `resolution=1.0` -- per `docs/changes.md`, the strongest baseline
available, beating even the original `cellstates`. Saves labels, `Theta`,
`phi`, and the DM log-likelihood to `baseline.npz`.

`pbmc3k_fit.py` (the isolated pyro venv) loads that, fits the
stick-breaking mixture at the *same* `Theta*phi` prior, and scores its own
hard partition with the identical eq. 15 formula
(`fastcellstates.model.DirichletMultinomial.cluster_loglik`, copied by hand
as `dm_total_loglik` rather than importing fastcellstates into this venv) --
a genuine apples-to-apples number, not a proxy. At this scale (G=13,714,
baseline finds 635/2,700 states live) the naive elementwise likelihood from
the synthetic script would materialise a (K, N, G) tensor; rewritten as a
single `(N,G) @ (G,K)` matmul computed once before the "cells" plate instead
(see the module docstring for why this is exactly equivalent under Pyro's
enumeration, not an approximation).

Result (`K_fit=700`, `gamma=5.0`, one seed, 10,000 SVI steps -- ELBO had
already plateaued by ~5,000, so this is a converged run, not an
undertrained one):

| | states | log-likelihood | time |
|---|---|---|---|
| baseline (`fast`, res=1.0) | 635 | -43,366,714.8 | 69.0s |
| pyro (stick-breaking) | 484 | -43,583,619.0 | 127.2s (10k steps) |

ARI between the two partitions: 0.33 (weak agreement). So on real data, at
this one untuned configuration, the SVI fit converges to a genuinely worse
optimum than the existing search -- about 0.5% lower log-likelihood, ~150
fewer states, and a different partition, not just a relabeling of the same
one. Not a proof against the idea (one `gamma`, one `K`, one seed, one
learning rate, plain mean-field, all picked without tuning), but the first
result that isn't favourable, and it's the one that matters more than the
synthetic checks: recovering your own generative model's synthetic data is
the easy case, beating (or even matching) the existing search on real data
is the actual bar.

Checked against the *cheapest* baseline too, not just the tuned one: the
plain `fast` preset (default resolution, no override) gets 26 states,
LL=-43,383,548.6, in 4.5s.

| | states | log-likelihood | time |
|---|---|---|---|
| `fast` (default) | 26 | -43,383,548.6 | 4.5s |
| pyro (stick-breaking) | 484 | -43,583,619.0 | 127.2s |
| `fast` (resolution=1.0) | 635 | -43,366,714.8 | 69.0s |

So the least-tuned thing the package already does beats the Pyro fit on
*both* axes at once: better log-likelihood, 24x fewer states, 28x less
time. This isn't the resolution-tuned baseline making the comparison look
unfairly hard -- the cheap default already dominates.
