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
`N_c`), fits a finite-`K` (over-truncated, flat Dirichlet on `pi` — not the
stick-breaking prior from the ideas note yet) mixture via enumeration + SVI,
and reports ARI against the true labels plus which states end up "live".
Simplest possible check: does the inference procedure itself recover known
clusters at all, before testing anything on real data or adding the DP prior.

First run (`K_true=4`, `G=50`, `N=500`, `K_fit=8`, 2000 SVI steps, default
seed): ~8s on CPU, 3/4 true states end up "live" (one absorbed into another),
ARI 0.86 against the true labels. So the enumeration + SVI machinery does
work end to end — next things to actually vary: seed/init sensitivity (is
the merged state a fluke of one run?), step count / learning rate, and only
then the stick-breaking prior from the ideas note.
