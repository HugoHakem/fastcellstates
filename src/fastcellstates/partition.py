"""
Initial partitions for the search.

- ``singletons``: every cell its own cluster (the paper's MCMC start).
- an **over-partition** of the PCA-kNN graph: Leiden (CPM or RBConfiguration)
  or walktrap cut to a target count.  It only has to *over*-segment; the
  DM-optimal merge finds the stopping point.  CPM's resolution is size-stable,
  RBConfiguration's is not (it has a resolution limit); walktrap-`cut_at(N/gamma)`
  targets a count directly.
"""

import numpy as np


def singletons(n):
    return np.arange(n, dtype=np.int32)


def _undirected_knn_graph(knn_idx):
    import igraph as ig

    n, k = knn_idx.shape
    src = np.repeat(np.arange(n), k)
    dst = knn_idx.ravel()
    e = np.unique(np.stack([np.minimum(src, dst), np.maximum(src, dst)], 1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    return ig.Graph(n=n, edges=e.tolist(), directed=False)


def leiden(knn_idx, resolution=0.1, objective="cpm", seed=1, n_iterations=2):
    """Leiden communities on the undirected kNN graph.

    objective: ``"cpm"`` (Constant Potts Model, absolute density threshold,
    size-stable) or ``"rbc"`` (RBConfiguration / modularity).
    """
    import leidenalg as la

    g = _undirected_knn_graph(knn_idx)
    cls = la.CPMVertexPartition if objective == "cpm" else la.RBConfigurationVertexPartition
    p = la.find_partition(
        g, cls, resolution_parameter=resolution, seed=seed, n_iterations=n_iterations
    )
    _, lab = np.unique(np.asarray(p.membership), return_inverse=True)
    return lab.astype(np.int32)


def walktrap_cut(knn_idx, n_groups, steps=4):
    """Walktrap dendrogram on the kNN graph, cut to ``n_groups`` communities."""
    g = _undirected_knn_graph(knn_idx)
    dend = g.community_walktrap(steps=steps)
    k = min(int(n_groups), g.vcount() - 1)
    _, lab = np.unique(np.asarray(dend.as_clustering(n=k).membership), return_inverse=True)
    return lab.astype(np.int32)


def over_partition(knn_idx, algorithm="cpm", resolution=0.1, gamma=35, n_groups=None, seed=1):
    """Dispatch.

    algorithm
        ``"cpm"``        : Leiden/CPM at ``resolution`` (default 0.1 ~ N/40)
        ``"leiden_rbc"`` : Leiden/RBConfiguration at ``resolution``
        ``"walktrap"``   : walktrap cut to ``n_groups`` (or ``round(N/gamma)``)
    """
    if algorithm == "walktrap":
        n = knn_idx.shape[0]
        k = n_groups if n_groups is not None else max(1, round(n / gamma))
        return walktrap_cut(knn_idx, k)
    obj = "cpm" if algorithm == "cpm" else "rbc"
    return leiden(knn_idx, resolution=resolution, objective=obj, seed=seed)
