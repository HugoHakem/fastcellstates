"""
kNN graph over cells: a *search heuristic* feeding two blocks:

- ``partition``: the Leiden / CPM / walktrap over-partition is built on this
  graph;
- ``moves``: the kNN-pruned deterministic sweep restricts each cell's
  candidate clusters to those of its neighbours (``MovesCfg.sweep_prune_k``).

The clustering objective and every ΔLL test stay the exact
Dirichlet-multinomial likelihood on raw counts, so a poor neighbour choice
only costs a missed candidate, never a wrong answer.  This keeps the "no data
pre-processing" property of the clustering itself while the *search* uses a
standard normalized-expression neighbourhood.

`cell_knn` is exact brute-force (fine up to ~50k cells with the PCA path).  Its
BLAS-heavy parts (PCA, the distance matmul) run under a scoped <=16-thread cap
(``_blas_limit``): OpenBLAS/MKL regress past that on these shapes.
``metric="sanity"`` is a different cost story entirely -- a compiled
(numba) O(N^2 * G) nested loop, not a BLAS matmul -- see ``sanity.py``.
"""

import os
from contextlib import contextmanager

import numpy as np
import scipy.sparse as sp


def _usable_cpus():
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@contextmanager
def _blas_limit(n=16):
    """Cap the BLAS pool for the enclosed block (scoped: never leaks).

    OpenBLAS / MKL regress past ~16 threads on the tall-skinny PCA and the
    brute-force kNN distance matmul here; the default is every core.  numba's
    own pool for the sweep is left alone.
    """
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=min(n, _usable_cpus()), user_api="blas"):
        yield


def _lognorm(data):
    """(G, N) counts -> (N, G) dense, median-library-size normalized, log1p'd."""
    X = data.T
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * np.median(lib))


def _pca_knn(X, k, n_pcs):
    """(N, G) dense features -> (N, k) int32 Euclidean kNN via PCA, self excluded."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    N = X.shape[0]
    k = min(k, N - 1)
    with _blas_limit():
        pcs = PCA(
            n_components=min(n_pcs, N - 1, X.shape[1]), svd_solver="randomized", random_state=0
        ).fit_transform(X)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(pcs)
        idx = nn.kneighbors(pcs, return_distance=False)  # (N, k+1), includes self
    # drop self (usually column 0, but be safe)
    out = np.empty((N, k), dtype=np.int32)
    for i in range(N):
        row = idx[i][idx[i] != i][:k]
        out[i] = row
    return out


def cell_knn(data, k=15, metric="pca", n_pcs=50):
    """k nearest neighbours of every cell.

    Parameters
    ----------
    data : (G, N) array or scipy.sparse, genes x cells UMI counts
    k : int
    metric : "pca" (default) | "cosine_log1p" | "sanity"
        "pca":         the classic Scanpy recipe: normalize_total, log1p,
                       z-score genes, PCA(n_pcs), Euclidean kNN on the scores.
        "cosine_log1p": cosine similarity on normalized log1p counts, all genes.
        "sanity":      Breda et al.'s Sanity (github.com/jmbreda/Sanity,
                       Nat. Biotechnol. 2021): per-gene empirical-Bayes fit
                       (see ``sanity.sanity_fit`` / ``_sanity_kernels``),
                       then its own uncertainty-weighted pairwise distance
                       (``sanity.sanity_distance``) -- no PCA.  A ported
                       nested-loop numba kernel, O(N^2 * G_kept): a
                       few-thousand-cell metric, unlike the other two.
    n_pcs : int, PCA components (metric="pca" only)

    Returns
    -------
    knn_idx : (N, k) int32, neighbour cell indices, nearest first, self excluded
    """
    if metric == "sanity":
        from . import sanity as _sanity

        D = _sanity.sanity_distance(_sanity.sanity_fit(data))
        N = D.shape[0]
        k = min(k, N - 1)
        np.fill_diagonal(D, np.inf)
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        row = np.arange(N)[:, None]
        order = np.argsort(D[row, part], axis=1)
        return part[row, order].astype(np.int32)

    X = _lognorm(data)
    N = X.shape[0]
    k = min(k, N - 1)

    if metric == "cosine_log1p":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        with _blas_limit():
            sim = Xn @ Xn.T
        np.fill_diagonal(sim, -np.inf)
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        row = np.arange(N)[:, None]
        order = np.argsort(-sim[row, part], axis=1)
        return part[row, order].astype(np.int32)

    if metric == "pca":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = np.clip((X - mu) / sd, -10, 10)  # scale (clipped)
        return _pca_knn(Xs, k, n_pcs)

    raise ValueError(f"unknown metric {metric!r}")
