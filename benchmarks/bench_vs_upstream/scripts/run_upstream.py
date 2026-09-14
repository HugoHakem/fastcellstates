"""
One original-cellstates run for the upstream comparison, replicating
scripts/run_cellstates.py's own default behaviour exactly (prior
optimization on, N_CACHE=10000, tries_per_step=1000), instrumented the same
way as run_fastcellstates.py. --threads defaults to 1 (their own CLI
default) but can be set higher: the original's OpenMP-parallel C extension
does respect it, it's just not on by default. Run inside the
bench_vs_upstream/upstream_cellstates pixi environment, which has the
original `cellstates` package installed.

    pixi run --manifest-path benchmarks/bench_vs_upstream/upstream_cellstates/pixi.toml python \
        benchmarks/bench_vs_upstream/scripts/run_upstream.py --threads 1 \
        --out benchmarks/bench_vs_upstream/results/upstream_t1.json
"""

import argparse
import json
import resource
import time

import numpy as np
from cellstates.cluster import Cluster
from cellstates.run import run_mcmc

DEFAULT_DATA = "benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy"
N_CACHE = 10_000
TRIES_PER_STEP = 1000
SEED = 1


def _fit_once(data, lam, cluster_init, threads):
    clst = Cluster(data, lam, cluster_init.copy(), num_threads=threads, n_cache=N_CACHE, seed=SEED)
    run_mcmc(clst, N_steps=clst.N_samples, tries_per_step=TRIES_PER_STEP, log_level="ERROR")
    return clst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="shared (G, N) .npy from prepare_data.py")
    ap.add_argument("--threads", type=int, default=1, help="scripts/run_cellstates.py's own default is 1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = np.load(args.data).astype(np.int64, copy=False)
    mask = np.any(data, axis=1)
    data = data[mask, :]
    G, N = data.shape

    t0 = time.perf_counter()

    alpha = 2.0 ** round(np.log2(data.sum() / N))
    lam = alpha * data.sum(axis=1) / data.sum()
    cluster_init = np.arange(N, dtype=np.int32)

    clst = _fit_once(data, lam, cluster_init, args.threads)

    # scripts/run_cellstates.py's own prior-optimization loop: probe alpha x2
    # (or /2) while it improves total_likelihood, resetting to the original
    # singleton partition and rerunning MCMC at each new alpha, until neither
    # direction improves on the current alpha.
    find_best_alpha = True
    while find_best_alpha:
        best_alpha, best_ll = alpha, clst.total_likelihood
        a = alpha
        while True:
            a *= 2
            clst.set_dirichlet_pseudocounts(a, n_cache=0)
            if clst.total_likelihood > best_ll:
                best_ll, best_alpha = clst.total_likelihood, a
            else:
                break
        if best_alpha == alpha:
            a = alpha
            while True:
                a /= 2
                clst.set_dirichlet_pseudocounts(a, n_cache=0)
                if clst.total_likelihood > best_ll:
                    best_ll, best_alpha = clst.total_likelihood, a
                else:
                    break
        clst.set_dirichlet_pseudocounts(best_alpha, n_cache=N_CACHE)
        if best_alpha != alpha:
            clst = _fit_once(data, best_alpha * data.sum(axis=1) / data.sum(), cluster_init, args.threads)
            alpha = best_alpha
        else:
            find_best_alpha = False

    secs = time.perf_counter() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    n_singletons = int((np.bincount(clst.clusters) == 1).sum())

    result = {
        "tool": "cellstates (upstream)",
        "config": f"threads={args.threads}",
        "n_states": int(clst.n_clusters),
        "n_singletons": n_singletons,
        "theta": float(alpha),
        "log_likelihood": float(clst.total_likelihood),
        "seconds": secs,
        "peak_rss_mb": peak_rss_mb,
        "threads": args.threads,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
