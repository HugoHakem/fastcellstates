"""Run under the project's real pixi env (needs fastcellstates itself, not
just pyro): produces the reference partition/likelihood the Pyro prototype
in pbmc3k_fit.py (run separately, in the isolated .venv-pyro) is scored
against. Kept as two scripts in two environments on purpose -- the pyro
venv stays light (no numba/fastcellstates), and this one needs no torch/pyro.

    pixi run python experiments/pyro_mixture/baseline.py
"""

import time

import numpy as np
import scipy.sparse as sp

import fastcellstates as fcs

DATA = "experiments/pyro_mixture/pbmc3k_counts.npy"
OUT = "experiments/pyro_mixture/baseline.npz"


def main():
    counts = sp.csc_matrix(np.load(DATA))
    # matches benchmarks/bench_vs_upstream's "fast_res1": docs/changes.md
    # reports this reaching the best log-likelihood of everything tested,
    # including the original cellstates -- the toughest available baseline.
    import copy

    cfg = copy.deepcopy(fcs.PRESETS["fast"])
    cfg.init.resolution = 1.0

    t0 = time.perf_counter()
    summ = fcs.run(counts, cfg)
    dt = time.perf_counter() - t0

    print(f"fast (resolution=1.0): {summ.n_states} states, LL={summ.log_likelihood:.1f}, {dt:.1f}s")
    np.savez(
        OUT,
        labels=summ.labels,
        theta=summ.theta,
        lam=summ.lam,
        log_likelihood=summ.log_likelihood,
        n_states=summ.n_states,
        seconds=dt,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
