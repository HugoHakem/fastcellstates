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
