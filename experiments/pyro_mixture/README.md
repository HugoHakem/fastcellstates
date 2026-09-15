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

### Why, and the cheap fix: letting Theta co-adapt

Candidate explanations, roughly by expected impact: (1) no data-informed
init -- every state starts at the same flat prior mean, `fast`/`exact` both
start from a real partition (Leiden or singletons); (2) no discrete merge
step -- the exact search explicitly merges any likelihood-improving pair,
mean-field only prunes weak states indirectly through the stick-breaking
prior's KL term, which plausibly explains ending up with far more live
states than even the cheap baseline needs; (3) mean-field is a strictly
weaker approximation than the exact search's direct (unfactorized)
optimization of the true marginal; (4) Theta was pinned to the *baseline's*
fitted value rather than allowed to adapt to Pyro's own (much worse)
partition -- a real confound, since `exact`/`fast` themselves treat Theta
and the partition as jointly fit; (5) plain Adam on Dirichlet/Beta
concentration params is a weaker optimizer geometry than the natural
gradients textbook CAVI would use for this exact conjugate family; (6)
objective/scoring mismatch -- SVI maximises the ELBO, not the hard-partition
score we read off afterward; (7) one seed, and mean-field mixtures are
known to be highly multimodal.

Fixed (4), the cheapest one: Theta is now a `pyro.param` point estimate,
optimized jointly by the same SVI step instead of pinned to the baseline's
value (phi stays fixed -- same free-scale-only treatment `exact`/`fast`
use). Result, same `K_fit=700`, `gamma=5.0`, one seed, 10,000 steps:

| | states | log-likelihood | Theta |
|---|---|---|---|
| pyro, Theta fixed | 484 | -43,583,619.0 | 10,568 (baseline's, pinned) |
| pyro, Theta co-adapted | 497 | -43,570,660.9 | 9,589 -> 10,568 (converged) |

A small, real improvement (~6% of the gap to the resolution=1.0 baseline
closed), not a game-changer -- consistent with (4) being a real but minor
factor, not the dominant one.

### The merge/sweep experiment: (2) was most of it

`merge_pyro_labels.py` (pixi env, real `fastcellstates`) takes the
Theta-co-adapted Pyro fit's raw hard labels (`pyro_fit.npz`) and runs them
through the actual pipeline polish: `Cluster._merge_clusters_optimally()`
then `moves.run_sweep` -- the exact same two steps `fast`/`exact` apply
after *their own* warm start (`pipeline.py`). (Also doubles as a check on
`dm_total_loglik`: `Cluster.total_likelihood` on the raw import matched the
pyro venv's own hand-computed number to the decimal, -43,570,660.9 both
ways.)

| | states | log-likelihood | gap to res=1.0 baseline |
|---|---|---|---|
| pyro, raw | 497 | -43,570,660.9 | -203,946.1 |
| pyro + merge | 223 | -43,501,654.4 | -134,939.6 |
| pyro + merge + sweep | 268 | -43,380,149.0 | **-13,434.2** |
| baseline (`fast`, res=1.0) | 635 | -43,366,714.8 | -- |
| baseline (`fast`, default) | 26 | -43,383,548.6 | -- |

Merge alone closes ~34% of the gap; merge+sweep closes ~93%, landing within
0.03% of the tuned baseline's log-likelihood -- and actually *beats* the
cheap default `fast` preset's -43,383,548.6, by 3,399.6, using 268 states
vs. its 26. So (2) (no discrete merge/reassignment mechanism) was most of
the story: the raw SVI partition wasn't fundamentally bad, it just needed
the exact same cleanup every warm start (Leiden included) already gets.

Caveats before reading too much into this: the merge/sweep step is cheap on
top of an already-expensive fit (Pyro itself: ~137s vs. the cheap
baseline's 4.5s -- the win above is quality, not cost, and total cost is
still ~30x the cheap baseline's); one seed; and this doesn't yet show SVI
adds anything Leiden doesn't -- it shows SVI's output, once handed to the
existing polish, is *roughly as good a starting point* as Leiden's. Whether
it's ever a *better* one (worth the extra ~130s) is still untested.

### Singleton-style init, and closed-form CAVI instead of Adam

Two follow-ups, agreed on before implementing (per the strategy discussion):
(1) attack cause (1) -- flat init -- with a real analogue of the MCMC's
singleton start: `K_fit = N = 2,700`, every state's `alpha_s` seeded from
one real cell's own counts (`init_tau_alpha(..., mode="real_cell")` in
`_common.py`, a straight bijection at `k == N`, no clustering algorithm
involved). (2) attack cause (5) -- Adam's optimizer geometry -- with actual
closed-form CAVI (`pbmc3k_cavi.py`): the model is fully conjugate (`z` was
*already* handled exactly by enumeration in the Adam version too -- that
doesn't change), so `alpha_s` and the stick weights `v_k` both have
closed-form coordinate updates given digamma-based expected sufficient
statistics (`E_q[log alpha]`, not `log(mean alpha)` -- the E-step
correctness detail flagged in the strategy discussion), no Pyro/Adam
needed. Theta gets the same two-way "try both" treatment: a few Adam steps
on `log(Theta)` each round, or a Brent line search on the same closed-form
objective each round (mirrors `fastcellstates.moves.log_search`'s approach
to fitting Theta for the exact/fast presets).

Four raw (pre-merge) runs, all `K_fit=2,700`, `gamma=5.0`:

| | init | optimizer | states | log-likelihood | gap | time |
|---|---|---|---|---|---|---|
| B | real_cell | Adam | 2,700 | -43,539,661.2 | -172,946.4 | 957.0s |
| C | flat | CAVI (gradient Theta) | 3 | -43,638,283.1 | -271,568.3 | 26.7s |
| D-gradient | real_cell | CAVI (gradient Theta) | 2,700 | -43,535,463.7 | -168,748.9 | 32.2s |
| D-linesearch | real_cell | CAVI (linesearch Theta) | 2,700 | -43,540,279.4 | -173,564.6 | 8.5s |

Three findings, none of them what I expected going in:

- **Flat init breaks CAVI specifically, and fast.** C collapses to 2-3
  states within ~10 rounds and stays there (Theta blows up to ~76,000,
  consistent with very coarse clusters). CAVI's updates are deterministic
  and monotonic -- no gradient noise to keep redundant components alive --
  so a fully symmetric start collapses much harder here than it did under
  Adam (which still had ~490 states after the same flat init). This is the
  cleanest confirmation yet that (1) and (5) interact: a bad init is worse
  under a *better* optimizer, not better.
- **Real-cell init stops the optimizer from doing much of anything.** All
  three real-cell-init runs (B, D-gradient, D-linesearch) land within a
  tight band (gap -169k to -174k) regardless of whether the continuous part
  is Adam or CAVI, gradient-Theta or linesearch-Theta -- and all three keep
  essentially every one of the 2,700 initial states alive. Once each state
  starts anchored to a real, distinct cell, neither optimizer does much
  active consolidating on its own; they mostly just refine each state
  in place. So init dominates here, optimizer choice is a rounding error on
  top of it -- the opposite of what motivated trying CAVI in the first
  place.
- **Theta: linesearch is smoother and ~4x faster, not obviously better.**
  Gradient-Theta oscillates in a small cycle (4650 <-> 4760, never quite
  settling); linesearch converges monotonically and finishes in a quarter
  the time (8.5s vs 32.2s) -- but the two final log-likelihoods differ by
  only 4,816 (statistically not much, one run each). Linesearch's cleaner
  convergence and lower cost make it the better default going forward
  regardless.

None of these four raw partitions are close to competitive yet, still worse
than even the *flat*-init Adam run from before (gap -203,946.1) -- the
singleton-heavy real-cell-init runs (B, D) sit at roughly 2,700 states, not
meaningfully different from where they started; nothing here does the
consolidation merge/sweep did before.

**Merge from ~2,700 starting states is far more expensive than from ~500**
(seconds before; B's took ~20-25 minutes at 2000%+ CPU -- genuinely
computing, not stuck, just a much-worse-than-linear cost in the number of
starting clusters that the earlier ~500-state run never exercised).

**And, once it finished: real-cell init does *worse* after merge+sweep than
flat init did**, despite avoiding the collapse problem --

| | states | log-likelihood | gap to res=1.0 baseline |
|---|---|---|---|
| B (real_cell, Adam), raw | 2,700 | -43,539,661.2 | -172,946.4 |
| B + merge | 432 | -43,451,995.0 | -85,280.2 |
| B + merge + sweep | 511 | -43,435,007.7 | **-68,292.9** |
| *(for comparison)* flat-init Adam + merge + sweep | 268 | -43,380,149.0 | *-13,434.2* |

Merge+sweep only closes ~61% of B's gap here, landing 5x further behind
than the earlier flat-init run did. Reading: flat init wasn't just "a bad
start Adam had to work around" -- the gradual, noisy training *from* a
shared symmetric start was itself doing real organizing work, similar cells
getting pulled toward shared states over thousands of SVI steps simply
because states that already attract similar cells reinforce each other via
the shared gradient signal. Real-cell init removes exactly that dynamic:
every state starts already fitting its own anchor cell well, so there's
little gradient pressure to reorganize anything, and the entire
consolidation job is left to one static greedy merge pass over ~2,700
near-arbitrary micro-clusters at once -- a harder job than consolidating an
already-partially-organized ~500-state partition.

**D-linesearch (real_cell, CAVI, linesearch Theta) tells a different story**
-- despite a raw gap almost identical to B's (-173,564.6 vs -172,946.4), it
responds to merge+sweep far better:

| | states | log-likelihood | gap to res=1.0 baseline |
|---|---|---|---|
| D-linesearch, raw | 2,700 | -43,540,279.4 | -173,564.6 |
| D-linesearch + merge | 511 | -43,405,261.6 | -38,546.8 |
| D-linesearch + merge + sweep | 590 | -43,391,882.2 | **-25,167.4** |

2.7x better than B's post-polish gap (-25,167.4 vs -68,292.9), from a raw
partition that scored about the same as B's before polish. So raw
log-likelihood alone doesn't predict how mergeable a partition is -- the
two fits apparently organize *which* cells are near which states
differently enough to matter a lot to a greedy hierarchical merge, even
while scoring almost identically as a whole. Whether this is "CAVI
organizes real-cell inits more usefully than Adam does" or something more
specific to this one run is the open question; scoring D-gradient (same
real-cell init, same CAVI E/M steps, only Theta's update rule differs) next
to see whether it lands near D-linesearch (implicating CAVI generally) or
nearer to B (implicating something specific to the linesearch Theta path).
