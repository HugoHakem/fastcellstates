"""
Compose the blocks into a run.

    io.read -> graph -> partition (init) -> model (DM) -> Cluster ->
    moves (mcmc / merge / sweep / Theta) -> Summary

``run(data, cfg)`` is the Python entry point; ``cli.main`` wraps it.  The
``fast`` / ``exact`` presets and the ``Config`` schema live in ``config.py``.
"""

import numpy as np
import scipy.sparse as sp

from . import io as _io
from . import moves as _moves
from . import partition as _part
from ._types import Counts
from .config import Config
from .core import Cluster
from .graph import cell_knn
from .summary import Summary


def _heuristic_theta(counts):
    tot = float(counts.sum())
    n = counts.shape[1]
    return 2.0 ** round(np.log2(tot / n))


def _initial_partition(cfg, knn):
    if cfg.init.source == "singletons":
        return None
    return _part.over_partition(
        knn,
        algorithm=cfg.init.algorithm,
        resolution=cfg.init.resolution,
        gamma=cfg.init.gamma,
        seed=cfg.seed,
    )


def _cluster_at(counts, theta, init_partition, cfg, move_knn=None, genes=None, phi=None):
    """One full clustering pass at a fixed Theta -> converged Cluster."""
    K0 = (int(init_partition.max()) + 1) if init_partition is not None else 0
    clst = Cluster(
        counts,
        float(theta),
        phi=phi,
        c=init_partition,
        genes=genes,
        max_clusters=K0,
        n_cache=cfg.model.n_cache,
        seed=cfg.seed,
    )
    if move_knn is not None:
        clst.set_move_knn(move_knn)
    if cfg.moves.mcmc:
        _moves.run_mcmc(
            clst, N_steps=clst.N_samples, tries_per_step=cfg.moves.mcmc_tries, log_level="ERROR"
        )
        return clst  # run_mcmc ends with merge + a sweep
    if cfg.moves.merge:
        clst._merge_clusters_optimally()
        clst.set_N_boxes(clst.n_clusters + 2)
    _moves.run_sweep(clst, method=cfg.moves.sweep, to_convergence=cfg.moves.sweep_to_convergence)
    clst.optimize_clusters()  # final merge + polish
    clst.set_N_boxes(clst.n_clusters)
    return clst


def run(data: str | list[str] | Counts, cfg: Config | None = None, genes=None, phi=None) -> Summary:
    """data: path(s) or a (G, N) counts array/matrix (genes x cells: the
    opposite of AnnData's own ``adata.X``, which is cells x genes; pass
    ``adata.X.T`` if building the array yourself, or pass the ``.h5ad`` path
    directly).  If the data is cells x genes and there's no convenient way to
    transpose it before calling (e.g. from the CLI), set ``cfg.transpose``
    instead.

    ``phi``: (G,) ndarray, optional -- pin the Dirichlet prior's direction
    (supp. info §A1 slot 5) instead of estimating it from ``data`` itself;
    Theta is untouched by this and still follows ``cfg.model.theta_method``
    (fixed / searched) as usual.  See ``core.Cluster``'s ``phi`` parameter
    and ``model.phi.global_phi``.  Python-API only -- an array has no clean
    CLI spelling yet.

    Returns a ``Summary``."""
    if cfg is None:
        cfg = Config()
    if cfg.n_threads > 0:
        import numba

        numba.set_num_threads(cfg.n_threads)
    if isinstance(data, (str, bytes)) or (
        isinstance(data, (list, tuple)) and data and isinstance(data[0], str)
    ):
        counts, genes, cells = _io.read(data)
        if cfg.transpose:
            genes = cells  # what were column headers are now the rows
    else:
        counts = data
    counts = sp.csc_matrix(counts) if sp.issparse(counts) else np.ascontiguousarray(counts)
    if cfg.transpose:
        counts = counts.T

    np.random.seed(cfg.seed)
    need_over = cfg.init.source == "over_partition"
    prune = cfg.moves.sweep_prune_k > 0 and cfg.init.source != "singletons"

    # one PCA-kNN, wide enough for both consumers (Leiden graph + sweep prune);
    # the graph never sees Theta so it is built once for the whole run.
    knn = move_knn = None
    if need_over or prune:
        kmax = max(cfg.graph.k if need_over else 0, cfg.moves.sweep_prune_k if prune else 0)
        knn_full = cell_knn(sp.csc_matrix(counts), k=kmax, metric="pca", n_pcs=cfg.graph.n_pcs)
        if need_over:
            knn = np.ascontiguousarray(knn_full[:, : cfg.graph.k], dtype=np.int32)
        if prune:
            move_knn = np.ascontiguousarray(knn_full[:, : cfg.moves.sweep_prune_k])

    init_partition = _initial_partition(cfg, knn)
    theta0 = cfg.model.theta or _heuristic_theta(counts)

    def build(t):
        return _cluster_at(counts, t, init_partition, cfg, move_knn, genes, phi)

    if cfg.model.theta_method == "fixed":
        clst = build(theta0)
    elif cfg.model.theta_method == "doubling":
        clst = _moves.doubling(build, theta0, max_evals=cfg.model.theta_rounds)[1]
    elif cfg.model.theta_method == "coordinate_ascent":
        clst = _moves.coordinate_ascent(
            build, theta0, max_evals=cfg.model.theta_rounds, tol=cfg.model.theta_tol
        )[1]
    elif cfg.model.theta_method == "log_search":
        clst = _moves.log_search(
            build, theta0, tol=cfg.model.theta_tol, max_evals=cfg.model.theta_rounds
        )[1]
    else:
        raise ValueError(f"unknown model.theta_method {cfg.model.theta_method!r}")

    # ``clst`` already carries the gene names, masked to the genes it kept
    # (Cluster drops all-zero-total genes), so summ.genes lines up with
    # summ.counts.  Don't re-attach the caller's unfiltered ``genes`` here.
    return Summary.from_cluster(clst, counts)
