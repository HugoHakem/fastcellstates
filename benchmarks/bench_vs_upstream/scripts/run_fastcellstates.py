"""
One fastcellstates run for the upstream comparison: loads the shared counts
matrix, runs the given config, times just the clustering call, and writes a
small JSON result.

The shared data is a dense .npy (the one format both tools read
identically), but fastcellstates is handed a sparse copy of it, matching
its actually-recommended usage (reading straight from .h5ad, never
densified) rather than the original's only mode. This isn't just cosmetic:
Cluster.__init__ rebuilds its internal CSC structure from scratch on every
Theta probe (log_search/doubling recluster from scratch each time), so a
dense array pays a full nonzero-scan on every single probe, while a sparse
one collapses that to a cheap no-op each time.

    pixi run python benchmarks/bench_vs_upstream/scripts/run_fastcellstates.py \
        --config exact --out benchmarks/bench_vs_upstream/results/exact.json
"""

import argparse
import copy
import json
import resource
import time

import numpy as np
import scipy.sparse as sp

import fastcellstates as fcs

DEFAULT_DATA = "benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy"

CONFIGS = {
    "exact": lambda: fcs.PRESETS["exact"],
    "fast": lambda: fcs.PRESETS["fast"],
    "fast_res1": lambda: _fast_at(1.0),
    "fast_log_search": lambda: _fast_theta("log_search"),
    "fast_coordinate_ascent": lambda: _fast_theta("coordinate_ascent"),
    "fast_doubling": lambda: _fast_theta("doubling"),
}


def _fast_at(resolution):
    cfg = copy.deepcopy(fcs.PRESETS["fast"])
    cfg.init.resolution = resolution
    return cfg


def _fast_theta(theta_method, theta_rounds=10):
    """The `fast` preset with `theta_method` swapped and `theta_rounds` (the
    shared max-evaluations cap across all three search strategies since
    moves.py's theta-search unification) held equal, so the three are
    compared at the same evaluation budget rather than each at its own
    historical default."""
    cfg = copy.deepcopy(fcs.PRESETS["fast"])
    cfg.model = fcs.ModelCfg(theta_method=theta_method, theta_rounds=theta_rounds)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--data", default=DEFAULT_DATA, help="shared (G, N) .npy from prepare_data.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    counts = sp.csc_matrix(np.load(args.data))
    cfg = CONFIGS[args.config]()

    t0 = time.perf_counter()
    summ = fcs.run(counts, cfg)
    secs = time.perf_counter() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    sizes = np.round(summ.weights * summ.n_cells).astype(int)
    result = {
        "tool": "fastcellstates",
        "config": args.config,
        "n_states": summ.n_states,
        "n_singletons": int((sizes == 1).sum()),
        "theta": summ.theta,
        "log_likelihood": summ.log_likelihood,
        "seconds": secs,
        "peak_rss_mb": peak_rss_mb,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
