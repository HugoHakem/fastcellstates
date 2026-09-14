"""Run under the project's real pixi env (needs fastcellstates itself, not
pyro): takes the Pyro fit's own raw hard partition (pyro_fit.npz, from
pbmc3k_fit.py) and runs it through the package's actual merge step --
the same `Cluster._merge_clusters_optimally()` the `fast`/`exact` pipeline
calls after its own warm start (pipeline.py).

Tests directly which story explains the gap to baseline.npz: "SVI's raw
partition is bad" (merge barely helps) vs. "SVI's raw partition just needed
the same cleanup every warm start needs" (merge closes most of the gap).

    pixi run python experiments/pyro_mixture/merge_pyro_labels.py
"""

import numpy as np
import scipy.sparse as sp

import fastcellstates as fcs

DATA = "experiments/pyro_mixture/pbmc3k_counts.npy"
BASELINE = "experiments/pyro_mixture/baseline.npz"
PYRO_FIT = "experiments/pyro_mixture/pyro_fit.npz"


def main():
    counts = sp.csc_matrix(np.load(DATA))
    base = np.load(BASELINE)
    pyro = np.load(PYRO_FIT)

    clst = fcs.Cluster(counts, l=float(pyro["theta"]), c=pyro["labels"].astype(np.int32))

    # sanity check: this should closely match pyro_fit.npz's own log_likelihood
    # (computed independently, by hand, in the pyro venv) -- confirms that
    # hand-copied formula agrees with the package's real one, and that l=theta
    # (a scalar) reconstructs the same phi profile used to fit theta in the
    # first place (both are a deterministic function of the same counts).
    print(f"pyro raw:   {clst.n_clusters} states, LL={clst.total_likelihood:.1f}  "
          f"(pyro_fit.npz says {float(pyro['log_likelihood']):.1f})")

    clst._merge_clusters_optimally()
    n_merge, ll_merge = clst.n_clusters, clst.total_likelihood
    print(f"pyro+merge: {n_merge} states, LL={ll_merge:.1f}")

    # the pipeline's other polish step: per-cell reassignment. merge can only
    # combine whole groups; this is what fixes individual mis-assigned cells.
    from fastcellstates import moves

    moves.run_sweep(clst, method="jacobi", to_convergence=True)
    n_sweep, ll_sweep = clst.n_clusters, clst.total_likelihood
    print(f"pyro+merge+sweep: {n_sweep} states, LL={ll_sweep:.1f}")

    base_ll = float(base["log_likelihood"])
    print(f"\nbaseline (fast, res=1.0): {int(base['n_states'])} states, LL={base_ll:.1f}")
    print(f"gap raw:         {float(pyro['log_likelihood']) - base_ll:.1f}")
    print(f"gap after merge: {ll_merge - base_ll:.1f}")
    print(f"gap after sweep: {ll_sweep - base_ll:.1f}")


if __name__ == "__main__":
    main()
