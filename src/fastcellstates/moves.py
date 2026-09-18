"""
Iterative search over the partition and the concentration ``Theta``, given a
fixed model (``model.DirichletMultinomial``).

Cell sweeps: greedy single-cell reassignment to a local likelihood maximum:
- ``gauss_seidel``: visit cells one at a time, apply each improving move
  immediately (the paper's ``optimize_cell_positions_simple``; iterated to
  convergence here).
- ``jacobi``: score every cell against the frozen state, then apply the
  improving ones ΔLL-descending with a cheap live-state recheck.  Converges to
  the same kind of fixed point as gauss_seidel (different move order, so not
  bit-identical), but the scoring phase is ``prange``, so it uses multiple
  cores where gauss_seidel can't.

MCMC: ``run_mcmc`` (from-singletons or warm-started).

Theta search: ``coordinate_ascent`` alternates re-clustering with the
model's ``fit_theta`` (was ``prior.py``); ``doubling`` and ``log_search``
instead probe ``total_likelihood`` directly at each Theta, reclustering from
scratch every probe (a coarse power-of-2 walk, or Brent's method).

The agglomerative DM-optimal merge is ``Cluster._merge_clusters_optimally``.
"""

import logging
from pathlib import Path

import numpy as np

from .model import _dm_kernels as _k


def gauss_seidel(clst, to_convergence=True, max_passes=40, tol=1.0):
    prev = clst.total_likelihood
    for _ in range(max_passes if to_convergence else 1):
        it = np.random.permutation(clst.N_samples).astype(np.int64)
        _k.optimize_cells(it, clst.cells, clst.state, clst.prior)
        clst.refresh_likelihood()
        if not to_convergence:
            break
        LL = clst.total_likelihood
        if prev + tol >= LL:
            break
        prev = LL


def jacobi(clst, recheck=1, max_passes=80, tol=1.0):
    N = clst.N_samples
    prop_c = np.zeros(N, dtype=np.int64)
    prop_delta = np.zeros(N, dtype=np.float64)
    prev = clst.total_likelihood
    for _ in range(max_passes):
        applied = _k.batch_sweep(clst.cells, clst.state, clst.prior, recheck, prop_c, prop_delta)
        clst.refresh_likelihood()
        LL = clst.total_likelihood
        if applied == 0 or abs(LL - prev) <= tol:
            break
        prev = LL


def run_sweep(clst, method="jacobi", to_convergence=True, recheck=1):
    if method == "gauss_seidel":
        gauss_seidel(clst, to_convergence=to_convergence)
    elif method == "jacobi":
        jacobi(clst, recheck=recheck, max_passes=80 if to_convergence else 1)
    elif method in (None, "none"):
        pass
    else:
        raise ValueError(f"unknown sweep method {method!r}")


# --------------------------------------------------------------------------- #
# MCMC driver (from-singletons or warm-started): breakpoints + tries escalation
# --------------------------------------------------------------------------- #


def run_mcmc(
    clst,
    results_dir=None,
    N_steps=10000,
    tries_per_step=1000,
    min_index=0,
    log_level="INFO",
    keep_intermediate=False,
):
    """
    function to run full Markov-chain Monte Carlo optimization algorithm on a
    Cluster object and save outputs in files.

    Parameters
    ----------
    clst : Cluster object
    results_dir : str, default=None
        path to directory where the latest intermediate cluster state is stored
        as clusters_intermediate.txt at each breakpoint.
        If None or empty string, nothing is saved.
    N_steps : int, default=10000
        Number of MCMC steps to initially run per breakpoint.
    tries_per_step : int, default=1000
        Number of moves initially proposed per step. The higher, the longer
        the optimization can last (and the better the optimum)
    log_level : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}, default='INFO'
        verbosity of information of progression. Set to 'ERROR' to turn off.
    keep_intermediate : bool, default=False
        all intermediate cluster states are saved as
        clusters_****.txt at each breakpoint (**** represents a 4-digit count).

    """
    logformat = "%(asctime)-15s - %(levelname)s:%(message)s"
    logging.basicConfig(format=logformat, level=getattr(logging, log_level))

    logging.debug(f"initially check output every {N_steps} steps")

    def _save(tag):
        if not results_dir:
            return
        name = f"clusters_{tag:04d}.txt" if keep_intermediate else "intermediate_clusters.txt"
        logging.debug(f"write output {tag:04d}")
        np.savetxt(Path(results_dir) / name, clst.clusters, fmt="%i")

    i = 0
    if results_dir:
        logging.debug(f"writing intermediate states to directory {results_dir}")
    _save(i)  # save initial configuration
    logging.debug(f"n_clusters={clst.n_clusters}, total likelihood={clst.total_likelihood}")
    old_likelihood = clst.total_likelihood
    best_clusters = clst.clusters.copy()

    while i < 10000:
        i += 1
        if clst.N_boxes != clst.n_clusters + 2:
            clst.set_N_boxes(clst.n_clusters + 2)
            logging.debug(f"changed N_boxes to {clst.N_boxes}")
        try:
            _ = clst.biased_monte_carlo_sampling(
                N_steps=N_steps, tries_per_step=tries_per_step, min_index=min_index
            )
        except RuntimeError as err:
            # N_batch = N_batch*10
            tries_per_step *= 10
            N_steps = N_steps // 10
            logging.debug(err)
            if N_steps < 10:
                logging.debug("MCMC converges, little further improvement expected")
                break
            else:
                logging.debug(
                    f"changed tries_per_step to {tries_per_step} and N_steps to {N_steps}"
                )
        finally:
            if clst.total_likelihood > old_likelihood:
                old_likelihood = clst.total_likelihood
                _save(i)
                best_clusters = clst.clusters.copy()
                logging.debug(
                    f"n_clusters={clst.n_clusters}, total likelihood={clst.total_likelihood}"
                )

            else:
                logging.debug("likelihood did not improve; clustering converged")
                # revert to better clustering

                clst.set_clusters(best_clusters)
                break

    logging.debug("optimize clusters.")
    clst.optimize_clusters()
    clst.set_N_boxes(clst.n_clusters)
    logging.debug("end of clustering")


# --------------------------------------------------------------------------- #
# Theta search: alternate (fit Theta) <-> (re-cluster).  Was prior.py.
# --------------------------------------------------------------------------- #


def _state_counts(clst):
    """(K, G) summed counts of the non-empty subsets of ``clst``."""
    C = np.asarray(clst.cluster_umi_counts.T, dtype=np.float64)
    return C[np.asarray(clst.cluster_sizes) > 0]


def coordinate_ascent(recluster, theta0, rounds=10, tol=0.02):
    """(fix Theta -> recluster) <-> (fix partition -> Minka MLE), to convergence.

    ``recluster(theta)`` -> a fresh, converged ``Cluster`` at that Theta.
    Each round reclusters from scratch (not from the previous round's
    partition), so the alternation is not guaranteed monotonic in the
    likelihood; ``rounds`` bounds the cost (a hard cap, not a target) and the
    best (theta, cluster) seen across all rounds is returned, mirroring
    ``doubling``, rather than whichever round happened to run last.
    """
    from .model import DirichletMultinomial

    theta = float(theta0)
    clst = recluster(theta)
    best_ll, best_theta, best_clst = clst.total_likelihood, theta, clst
    for _ in range(rounds):
        new = DirichletMultinomial(theta, clst.phi).fit_theta(_state_counts(clst)).theta
        converged = abs(np.log(new) - np.log(theta)) < tol
        theta = new
        clst = recluster(theta)
        if clst.total_likelihood > best_ll:
            best_ll, best_theta, best_clst = clst.total_likelihood, theta, clst
        if converged:
            break
    return best_theta, best_clst


def log_search(recluster, theta0, tol=0.01, max_evals=30):
    """Maximise total_likelihood over Theta by Brent's method on log(Theta)
    (``scipy.optimize.minimize_scalar``): golden-section bracketing plus
    parabolic interpolation, rather than ``doubling``'s coarse power-of-2
    grid.  Assumes the likelihood is unimodal in log(Theta) (checked
    empirically on real data, not enforced here); every probe reclusters
    from scratch, like ``doubling``, so nothing is carried across probes.
    Returns ``(theta_star, cluster)``.
    """
    from scipy.optimize import minimize_scalar

    cache: dict[float, tuple[float, object]] = {}

    def neg_ll(x):
        if x not in cache:
            clst = recluster(float(np.exp(x)))
            cache[x] = (clst.total_likelihood, clst)
        return -cache[x][0]

    x0 = float(np.log(theta0))
    minimize_scalar(
        neg_ll,
        bracket=(x0 - np.log(2.0), x0 + np.log(2.0)),
        method="brent",
        options={"xtol": tol, "maxiter": max_evals},
    )
    best_x = max(cache, key=lambda x: cache[x][0])
    _, best_clst = cache[best_x]
    return float(np.exp(best_x)), best_clst


def doubling(recluster, theta0):
    """The original prior search: probe Theta x2, then /2; adopt the better and
    re-cluster once each way.  One step per direction, not a full optimum.
    Returns ``(theta_star, cluster)``.
    """
    theta = float(theta0)
    clst = recluster(theta)
    best_ll, best_theta, best_clst = clst.total_likelihood, theta, clst
    for factor in (2.0, 0.5):
        t = theta * factor
        while True:
            c = recluster(t)
            if c.total_likelihood > best_ll:
                best_ll, best_theta, best_clst = c.total_likelihood, t, c
                t *= factor
            else:
                break
        if best_theta != theta:
            break
    return best_theta, best_clst
