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

Theta search: all three strategies share the same skeleton -- propose a
candidate Theta, recluster fully at it, track the best (Theta, cluster) seen
so far (:func:`_recluster_and_track`) -- and differ only in how the next
Theta is proposed. ``coordinate_ascent`` and ``doubling`` propose from the
current state alone (Minka's exact best-response, or a two-phase greedy
x2/x0.5 walk) and share ``_recluster_and_track``'s bookkeeping directly;
``log_search`` instead hands the problem to Brent's method
(``scipy.optimize.minimize_scalar``), which needs the *history* of every
probe to decide its next one, so it keeps its own cache rather than fitting
the shared helper's per-call shape.

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


def _recluster_and_track(recluster, theta, best):
    """One (theta, recluster) probe, shared by ``coordinate_ascent`` and
    ``doubling``: recluster at ``theta``, then compare against ``best`` (a
    ``(best_ll, best_theta, best_clst)`` triple). Reclustering starts over
    from the same initial partition every time (never warm-started from a
    previous probe's result -- the search explored at one Theta shouldn't
    depend on what a different Theta's search happened to converge to), so
    it is not guaranteed to improve on the previous probe; every theta
    search here therefore tracks the best pair seen across all probes rather
    than trusting whichever one ran last.

    Returns ``(clst, best, improved)``.
    """
    clst = recluster(theta)
    if clst.total_likelihood > best[0]:
        return clst, (clst.total_likelihood, theta, clst), True
    return clst, best, False


def coordinate_ascent(recluster, theta0, max_evals=10, tol=0.02):
    """(fix Theta -> recluster) <-> (fix partition -> Minka MLE), to convergence.

    ``recluster(theta)`` -> a fresh, converged ``Cluster`` at that Theta.
    Each round reclusters from scratch (see :func:`_recluster_and_track`), so
    the alternation is not guaranteed monotonic in the likelihood;
    ``max_evals`` bounds the cost (a hard cap, not a target) and the best
    (theta, cluster) seen across all rounds is returned, mirroring
    ``doubling`` / ``log_search``, rather than whichever round happened to
    run last. ``tol``: stop once Minka's next proposed Theta barely moves
    ``log(theta)`` from the current round's -- "has the alternation stopped
    moving," not the same notion as ``log_search``'s ``tol`` (Brent's own
    bracket-width tolerance).
    """
    from .model import DirichletMultinomial

    theta = float(theta0)
    clst = recluster(theta)
    best = (clst.total_likelihood, theta, clst)
    for _ in range(max_evals):
        new = DirichletMultinomial(theta, clst.phi).fit_theta(_state_counts(clst)).theta
        converged = abs(np.log(new) - np.log(theta)) < tol
        theta = new
        _, best, _ = _recluster_and_track(recluster, theta, best)
        if converged:
            break
    return best[1], best[2]


def log_search(recluster, theta0, tol=0.01, max_evals=30):
    """Maximise total_likelihood over Theta by Brent's method on log(Theta)
    (``scipy.optimize.minimize_scalar``): golden-section bracketing plus
    parabolic interpolation, rather than ``doubling``'s coarse power-of-2
    grid or ``coordinate_ascent``'s Minka-guided proposals. Structurally
    different from those two, not just a different step rule: Brent decides
    its next probe from the *history* of every point evaluated so far (a
    parabola through the recent points), where the other two only ever need
    the current state -- so it isn't expressible as a per-call "propose the
    next theta" step the way theirs are. Assumes the likelihood is unimodal
    in log(Theta) (checked empirically on real data, not enforced here);
    every probe reclusters from scratch, like the other two, so nothing is
    carried across probes. Because unimodality isn't guaranteed, every probe
    is cached and the best-scoring one is returned, rather than trusting
    whichever point Brent itself converged to -- same "track the best seen,
    not the last one" idea as :func:`_recluster_and_track`, just via a cache
    instead of a running triple. ``tol`` is Brent's own bracket-width
    tolerance on log(Theta) (``xtol``) -- not the same notion as
    ``coordinate_ascent``'s ``tol`` (has the alternation's proposed Theta
    stopped moving). ``max_evals``: same name and same role (a hard cap on
    reclusters) as ``coordinate_ascent``'s, but not necessarily the same
    value -- Minka's step is an informed best-response and often needs few
    rounds; Brent's steps are comparatively blind (each only narrows a
    bracket), so a fair comparison may need a larger budget here, not a
    shared default. Returns ``(theta_star, cluster)``.
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


def doubling(recluster, theta0, max_evals=10):
    """The original prior search: probe Theta x2 (or /2), adopting and
    re-clustering again in whichever direction keeps improving; stops the
    moment a probe fails to improve, without trying the other direction once
    the first has moved at all. Legacy: reproduces the original published
    algorithm's exact search, not a general optimum -- optimal only up to a
    factor of 2, and it never bisects inside a gap. Uses the same
    "recluster, then track the best (theta, cluster) seen" idea as
    :func:`_recluster_and_track` / ``coordinate_ascent``, just with a
    two-phase greedy proposal (keep doubling/halving while it helps) instead
    of Minka's best-response. ``max_evals`` is a safety cap on the total
    number of reclusters across both directions -- the original algorithm has
    no such cap (unbounded in principle if pathological data kept improving
    indefinitely); in the ordinary case this rarely binds, since the search
    self-terminates in a handful of probes.
    """
    theta = float(theta0)
    clst = recluster(theta)
    best = (clst.total_likelihood, theta, clst)
    evals = 0
    for factor in (2.0, 0.5):
        t = theta * factor
        while evals < max_evals:
            evals += 1
            _, best, improved = _recluster_and_track(recluster, t, best)
            if not improved:
                break
            t *= factor
        if best[1] != theta:
            break
    return best[1], best[2]
