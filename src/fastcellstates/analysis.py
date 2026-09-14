"""
Interpret a clustering result: the merge hierarchy (DataFrame / Newick / scipy
linkage / re-cut), per-gene merge contributions, and marker-gene scores for
each split.  All of it is a function of the K per-state count vectors; see
``Summary.hierarchy`` / ``.cut`` / ``.markers``.
"""

import numpy as np
import pandas as pd
from numba import njit
from scipy.special import betainc, gammaln, logit

from .model._dm_kernels import find_cluster_distance

# ------ cluster hierarchies ------


def get_hierarchy_df(cluster_hierarchy, delta_LL_history):
    """
    create pandas DataFrame from output of get_cluster_hierarchy method
    of Cluster class object.
    """
    hierarchy_df = pd.DataFrame(
        columns=["cluster_new", "cluster_old", "delta_LL"], index=np.arange(len(cluster_hierarchy))
    )
    hierarchy_df.loc[:, ["cluster_new", "cluster_old"]] = np.array(cluster_hierarchy)
    hierarchy_df.loc[:, "delta_LL"] = delta_LL_history

    return hierarchy_df


def hierarchy_to_newick(hierarchy_df, clusters, cell_leaves=True, distance=True, min_distance=0.0):
    """
    Function for getting a newick string from a hierarchy DataFrame.

    Parameters
    ----------
    hierarchy_df : DataFrame containing cluster merges
    clusters : numpy array, default=None
        initial cluster configuration.
    cell_leaves : bool, default=True
        whether to include cells as leaves; otherwise clusters are leaves
    distance : bool, default=True
        whether to include distance (negative change in log-likelihood)
        in newick tree
    min_distance : float, default=0.
        minimal branch length for very small or positive changes in
        log-likelihood

    Returns
    -------
    newick_string : str
        string of cluster hierarchy in Newick format
    """
    cluster_names = np.unique(clusters)
    cluster_string_dict = {}
    cluster_distance = dict.fromkeys(cluster_names, min_distance)
    for c in cluster_names:
        if cell_leaves:
            cluster_args = np.argwhere(clusters == c).flatten().astype(str)
            if distance:
                cluster_string = (
                    "(" + (f":{min_distance},").join(cluster_args) + f":{min_distance})C{c}"
                )
            else:
                cluster_string = "(" + ",".join(cluster_args) + f")C{c}"
        else:
            cluster_string = f"C{c}"

        cluster_string_dict[c] = cluster_string

    c_low = min(cluster_string_dict.keys())

    distances = np.array([])
    if distance:
        distances = np.cumsum(
            np.where(
                (hierarchy_df.delta_LL >= 0), min_distance, -hierarchy_df.delta_LL + min_distance
            )
        )

    # use a running counter i instead of the frame index in case it is non-standard
    i = hierarchy_df.shape[0] - 1
    for _, row in hierarchy_df.iterrows():
        c_old = row.cluster_old
        c_new = row.cluster_new
        s_old = cluster_string_dict[c_old]
        s_new = cluster_string_dict[c_new]
        if distance:
            d = float(distances[-i - 1])
            d_old = cluster_distance[c_old]
            d_new = cluster_distance[c_new]
            cluster_string_new = f"({s_new}:{d - d_new},{s_old}:{d - d_old})I{i}"
            cluster_distance[c_new] = d
        else:
            cluster_string_new = f"({s_new},{s_old})I{i}"

        cluster_string_dict[c_new] = cluster_string_new
        del cluster_string_dict[c_old]
        del cluster_distance[c_old]

        i -= 1

    newick_string = cluster_string_dict[c_low] + ";"
    return newick_string


def get_scipy_hierarchy(hierarchy_df, return_labels=False):
    """
    function to get scipy.cluster.hierarchy linkage matrix

    Parameters
    ----------
    hierarchy_df : DataFrame containing cluster merges
    return_labels : whether to return leaf labels

    Returns
    -------
    Z : ndarray
        scipy linkage matrix
    labels : 1D array, optional
        leaf labels that can be used in scipy dendrogram
    """
    N_steps = hierarchy_df.shape[0]
    delta_LL_history = -hierarchy_df.delta_LL.values

    min_delta_LL = np.min(delta_LL_history)
    if min_delta_LL < 0:
        # need to renormalize to only have positive values
        delta_LL_offset = -min_delta_LL
    else:
        delta_LL_offset = 0.0

    Z = np.zeros((N_steps, 4))
    Z[:, 2] = delta_LL_history + delta_LL_offset
    cluster_names = np.unique(hierarchy_df.iloc[:, :2]).astype(int)
    clusterindex = dict(zip(cluster_names, range(N_steps + 1)))
    # Z[:, 3] = size of the newly formed cluster (scipy's linkage convention):
    # every leaf starts at size 1, not at its rank.  A copy-paste of
    # clusterindex's `range(N_steps + 1)` above, inherited from upstream.
    clustersize = dict.fromkeys(cluster_names, 1)
    for i, row in hierarchy_df.iterrows():
        idx_old, idx_new = int(row.cluster_old), int(row.cluster_new)
        cs = clustersize[idx_old] + clustersize[idx_new]
        clustersize[idx_new] = cs
        clustersize[idx_old] = 0
        Z[i, 3] = cs

        Z[i, 0] = min(clusterindex[idx_old], clusterindex[idx_new])
        Z[i, 1] = max(clusterindex[idx_old], clusterindex[idx_new])

        clusterindex[idx_new] = N_steps + 1 + i
        clusterindex[idx_old] = -1

    if return_labels:
        return Z, cluster_names
    else:
        return Z


def clusters_from_hierarchy(hierarchy_df, cluster_init=None, steps=-1):
    """
    Get merged clusters from hierarchy_df
    hierarchy_df : DataFrame containing cluster merges
    cluster_init : numpy array, default=None
        initial cluster configuration.
        If None, use np.arange(N+1) where N is size of hierarchy
    steps : int, default=-1
        Number of merging steps; if negative perform N+steps steps.
        E.g. with steps=-1 all except the last merge are performed resulting
        in 2 clusters.
    """
    N = hierarchy_df.shape[0]
    if steps < 0:
        steps = N + steps
    clusters = np.arange(N + 1) if cluster_init is None else cluster_init.copy()
    for step in range(steps):
        line = hierarchy_df.iloc[step]
        c_old = line["cluster_old"]
        c_new = line["cluster_new"]
        clusters[clusters == c_old] = c_new
    return clusters


# ------ functions for finding marker genes ------


def binomial_p(n, lam):
    """log P(n | N, Theta) per gene, the Beta-binomial marker score building
    block (supp. info eq. 37).  ``n``, ``lam`` are (G,) arrays: a subset's
    summed counts and its Dirichlet pseudocounts; N and Theta are read off as
    ``n.sum()`` / ``lam.sum()``."""
    lam_sum = np.sum(lam)
    n_sum = np.sum(n)
    P = (
        gammaln(lam_sum)
        - gammaln(lam)
        - gammaln(lam_sum - lam)
        + gammaln(n + lam)
        + gammaln(n_sum + lam_sum - n - lam)
        - gammaln(n_sum + lam_sum)
    )

    return P


def gene_contribution(n1, n2, lam):
    """Per-gene log-likelihood change from merging two subsets (n1, n2 -> n1+n2)."""
    d = binomial_p(n1 + n2, lam) - binomial_p(n1, lam) - binomial_p(n2, lam)
    return d


def gene_contribution_multi(all_n, lam):
    """Per-gene log-likelihood change from merging a list of subsets into one."""
    d = 0
    all_n_sum = np.zeros_like(all_n[0])
    for n in all_n:
        d -= binomial_p(n, lam)
        all_n_sum += n
    d += binomial_p(all_n_sum, lam)
    return d


def gene_contribution_table(clst, hierarchy_df):
    """
    Returns a table that, for each step in the cluster hierarchy, quantifies
    how much each gene contributes to the change in log-likelihood when these
    clusters are merged. In other words, it gives a score for each gene for
    how different its mean expression is between branches.

    Parameters
    ----------
    clst : Cluster
    hierarchy_df : hierarchy DataFrame of clst

    Returns
    -------
    score_table : (N_merges, N_genes) numpy array of floats
        each row corresponds to a row in hierarchy_df, each column to a gene.
        Values indicate single gene contributions to change in log-likelihood
        of two clusters being merged - large negative values are marker genes
    """
    orignal_clusters = clst.clusters.copy()

    score_table = np.zeros((hierarchy_df.shape[0], clst.G))
    for i, row in hierarchy_df.iterrows():
        c_old, c_new = int(row.cluster_old), int(row.cluster_new)
        d = gene_contribution(
            clst.cluster_umi_counts[:, c_old],
            clst.cluster_umi_counts[:, c_new],
            clst.dirichlet_pseudocounts,
        )
        score_table[i, :] = d

        clst.combine_two_clusters(c_new, c_old)
    clst.set_clusters(orignal_clusters)

    return score_table


def marker_score_table(clst, hierarchy_df):
    """
    Get marker gene scores for each step in a cluster hierarchy.

    Parameters
    ----------
    clst : Cluster
    hierarchy_df : hierarchy DataFrame of clst

    Returns
    -------
    marker_table : (N_merges, N_genes) numpy array of floats
        each row corresponds to a row in hierarchy_df, each column to a gene.
        Values indicate single gene contributions to change in log-likelihood
        of two clusters being merged - large negative values are marker genes
    """
    # element i is the list of cell-states currently folded into box i;
    # initially just [i] itself, growing as the hierarchy replays merges
    cellstate_clusters = []
    for i in range(clst.N_boxes):
        if clst.cluster_sizes[i]:
            cellstate_clusters.append([i])
        else:
            cellstate_clusters.append([])

    score_table = np.zeros((hierarchy_df.shape[0], clst.G))
    for i, row in hierarchy_df.iterrows():
        c_old, c_new = int(row.cluster_old), int(row.cluster_new)
        d = marker_scores(clst, cellstate_clusters[c_new], cellstate_clusters[c_old])
        score_table[i, :] = d

        cellstate_clusters[c_new].extend(cellstate_clusters[c_old])

    return score_table


# ------ marker genes + pairwise merge costs ------


def marker_scores(clst, C1, C2):
    """Marker-gene scores separating cell-state groups C1 and C2 (supp. info
    §A4): for each (c1, c2) pair we add
    ``betainc(n_gc1 + lambda_g, n_gc2 + lambda_g, x) * weight`` per gene, with
    ``x = (N_c1 + Theta) / (N_c1 + N_c2 + 2 Theta)`` and
    ``weight = |c1| |c2| / (sum|C1| sum|C2|)``, then take the logit.
    Positive score => gene higher in C2 than C1.
    """
    C1 = np.asarray(list(C1), dtype=np.int64)
    C2 = np.asarray(list(C2), dtype=np.int64)

    lam = np.asarray(clst.dirichlet_pseudocounts, dtype=np.float64)
    lam_sum = float(lam.sum())
    counts = np.asarray(clst.cluster_umi_counts, dtype=np.float64)
    umi_sum = np.asarray(clst.cluster_umi_sum, dtype=np.float64)
    sizes = np.asarray(clst.cluster_sizes, dtype=np.float64)

    C1C2 = sizes[C1].sum() * sizes[C2].sum()
    gene_scores = np.zeros(int(clst.G), dtype=np.float64)

    for c1 in C1:
        a = counts[:, c1] + lam
        n1 = umi_sum[c1] + lam_sum
        for c2 in C2:
            b = counts[:, c2] + lam
            x = n1 / (umi_sum[c1] + umi_sum[c2] + 2.0 * lam_sum)
            weight = sizes[c1] * sizes[c2] / C1C2
            gene_scores += betainc(a, b, x) * weight

    return logit(gene_scores)


@njit(cache=True)
def _all_cluster_distances(state, prior):
    n_boxes = state.sizes.shape[0]
    X = np.zeros(n_boxes * (n_boxes - 1) // 2, dtype=np.float64)
    idx = 0
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            if state.sizes[i] > 0 and state.sizes[j] > 0:
                X[idx] = find_cluster_distance(i, j, state, prior)
            idx += 1
    return X


def get_cluster_distances(clst):
    """Condensed vector of the change in total LL for merging every cluster pair
    (0 for pairs where either cluster is empty).  Length n_boxes*(n_boxes-1)/2.
    """
    return _all_cluster_distances(clst.state, clst.prior)
