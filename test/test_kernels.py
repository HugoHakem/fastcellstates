"""
Kernel correctness for the numba backend.

The Cython backend is gone; deterministic results are now checked against
(a) `fastcellstates.model._dm_reference.partition_loglik` (via the `oracle` shim),
    a plain-numpy DM marginal likelihood, and
(b) `test/data/cython_golden.npz`, values frozen from the last Cython build.

Run:  pixi run pytest test/test_kernels.py -q
"""

import contextlib
from pathlib import Path

import numpy as np
import pytest
from oracle import partition_loglik

from fastcellstates.core import Cluster

GOLD = np.load(Path(__file__).parent / "data" / "cython_golden.npz")
DATA = GOLD["data"]
LABELS = GOLD["labels"]
OVER = GOLD["over_labels"]
GMASK = DATA.sum(axis=1) > 0  # the Cluster drops all-zero genes
DATA_M = DATA[GMASK]  # counts on the kept genes (matches nb.G / nb.dirichlet_pseudocounts)


@pytest.fixture(scope="module")
def toy():
    return DATA, LABELS


# --------------------------------------------------------------------------- #
# deterministic numerics vs the numpy oracle + frozen Cython golden values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [64.0, 256.0, 1024.0])
def test_init_likelihood_matches_oracle(toy, alpha):
    data, _ = toy
    nb = Cluster(data, alpha, n_cache=200)
    tot, _ = partition_loglik(DATA_M, np.arange(data.shape[1]), nb.dirichlet_pseudocounts)
    np.testing.assert_allclose(nb.total_likelihood, tot, rtol=1e-9)
    np.testing.assert_allclose(
        nb.dirichlet_pseudocounts, GOLD[f"init_pseudocounts_{alpha}"], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.sort(nb.likelihood), np.sort(GOLD[f"init_like_{alpha}"]), rtol=1e-10
    )
    np.testing.assert_allclose(nb.total_likelihood, GOLD[f"init_totLL_{alpha}"], rtol=1e-10)


def test_shared_partition_likelihood(toy):
    data, labels = toy
    nb = Cluster(data, 256.0, c=labels.copy(), n_cache=200)
    tot, per = partition_loglik(DATA_M, labels, nb.dirichlet_pseudocounts)
    np.testing.assert_allclose(nb.total_likelihood, tot, rtol=1e-9)
    for c in range(3):
        np.testing.assert_allclose(nb.likelihood[c], per[c], rtol=1e-9)


def test_merge_delta_matches_oracle(toy):
    data, labels = toy
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        nb = Cluster(data, 256.0, c=labels.copy(), n_cache=200)
        ll0 = nb.total_likelihood
        nb.combine_two_clusters(i, j)
        got = nb.total_likelihood - ll0
        merged = labels.copy()
        merged[merged == j] = i
        want = (
            partition_loglik(DATA_M, merged, nb.dirichlet_pseudocounts)[0]
            - partition_loglik(DATA_M, labels, nb.dirichlet_pseudocounts)[0]
        )
        np.testing.assert_allclose(got, want, rtol=1e-8)


def test_get_best_move_matches_golden(toy):
    data, labels = toy
    nb = Cluster(data, 256.0, c=labels.copy(), n_cache=200)
    moves = np.array([nb.get_best_move(m)[0] for m in range(data.shape[1])])
    deltas = np.array([nb.get_best_move(m)[1] for m in range(data.shape[1])])
    np.testing.assert_array_equal(moves, GOLD["best_moves_256"])
    np.testing.assert_allclose(deltas, GOLD["best_deltas_256"], rtol=1e-8)


def test_merge_hierarchy_matches_golden():
    nb = Cluster(DATA, 256.0, c=OVER.copy(), n_cache=300)
    h, d = nb.merge_clusters(LL_threshold=-np.inf, n_cluster_threshold=1)
    np.testing.assert_array_equal(np.array(h), GOLD["merge_hier_256"])
    np.testing.assert_allclose(np.asarray(d), GOLD["merge_delta_256"], rtol=1e-8)


def test_merge_hierarchy_matches_bruteforce_reference():
    """The prange + row_best-cache merge must reproduce, step for step, a naive
    O(K^3) greedy on the pure-numpy DM likelihood: same pair, same order, same
    ΔLL, for a partition with many small clusters (exercises the cache
    invalidation paths that the tiny golden case does not)."""
    from fastcellstates.model._dm_reference import partition_loglik as _pl

    rng = np.random.default_rng(4)
    G, K_true, per = 120, 8, 9
    phi = rng.dirichlet(np.ones(G))
    fc = rng.dirichlet(400 * phi, size=K_true)
    cols = [rng.multinomial(600, fc[c]) for c in range(K_true) for _ in range(per)]
    data = np.array(cols).T  # (G, K_true*per)
    K0 = data.shape[1]
    over = np.arange(K0, dtype=np.int32)  # every cell its own box

    nb = Cluster(data, 256.0, c=over.copy(), max_clusters=K0, n_cache=400)
    lam_kept = nb.dirichlet_pseudocounts
    got_h, got_d = nb.merge_clusters(LL_threshold=0.0, n_cluster_threshold=1)
    got_labels = nb.clusters.copy()  # exercises the deferred (union-find) relabel

    # brute-force reference: same greedy, O(K^3), on the pure-numpy DM likelihood
    counts = np.asarray(nb.umi_data)
    labels = over.copy().astype(np.int64)
    ref_h, ref_d = [], []
    tot0, _ = _pl(counts, labels, lam_kept)
    while len(np.unique(labels)) > 1:
        live = sorted(int(x) for x in np.unique(labels))
        best = None
        for a in range(len(live)):
            for b in range(a):
                i, j = live[a], live[b]  # i > j
                merged = labels.copy()
                merged[merged == i] = j
                d = _pl(counts, merged, lam_kept)[0] - tot0
                if best is None or d > best[0]:
                    best = (d, i, j)
        assert best is not None
        if best[0] <= 0.0:  # LL_threshold
            break
        d, i, j = best
        labels[labels == i] = j
        tot0 += d
        ref_h.append([j, i])
        ref_d.append(d)

    np.testing.assert_array_equal(np.array(got_h), np.array(ref_h))
    np.testing.assert_allclose(np.asarray(got_d), np.asarray(ref_d), rtol=1e-7)
    hom, com = _hom_com(got_labels, labels)  # same partition (relabel-invariant)
    assert hom == 1.0 and com == 1.0


def test_set_dirichlet_pseudocounts_matches_golden(toy):
    data, labels = toy
    nb = Cluster(data, 128.0, c=labels.copy(), n_cache=300)
    nb.set_dirichlet_pseudocounts(512.0, n_cache=0)
    np.testing.assert_allclose(np.sort(nb.likelihood), np.sort(GOLD["setdir_like_512"]), rtol=1e-10)
    np.testing.assert_allclose(nb.total_likelihood, GOLD["setdir_totLL_512"], rtol=1e-10)
    tot, _ = partition_loglik(DATA_M, labels, nb.dirichlet_pseudocounts)
    np.testing.assert_allclose(nb.total_likelihood, tot, rtol=1e-9)


def test_marker_scores_matches_golden(toy):
    from fastcellstates.analysis import marker_scores

    data, labels = toy
    nb = Cluster(data, 256.0, c=labels.copy(), n_cache=300)
    np.testing.assert_allclose(
        marker_scores(nb, [0], [1]), GOLD["marker_0_1"], rtol=1e-6, atol=1e-8, equal_nan=True
    )
    np.testing.assert_allclose(
        marker_scores(nb, [0, 1], [2]), GOLD["marker_01_2"], rtol=1e-6, atol=1e-8, equal_nan=True
    )


def test_cluster_distances_matches_golden(toy):
    from fastcellstates.analysis import get_cluster_distances

    data, labels = toy
    nb = Cluster(data, 256.0, c=labels.copy(), n_cache=300)
    np.testing.assert_allclose(
        np.asarray(get_cluster_distances(nb)), GOLD["cluster_distances"], rtol=1e-8
    )


@pytest.mark.parametrize("method", ["jacobi", "gauss_seidel"])
def test_knn_pruned_sweep_matches_full(method):
    """With a wide kNN (every cluster reachable) the pruned sweep must reach the
    same fixed point as the full scan: same ΔLL test, just fewer candidates."""
    from fastcellstates.graph import cell_knn
    from fastcellstates.moves import run_sweep

    knn = cell_knn(np.ascontiguousarray(DATA_M), k=min(150, DATA_M.shape[1] - 1), n_pcs=20)

    def sweep(pruned):
        c = Cluster(DATA, 256.0, c=OVER.copy(), n_cache=300, seed=1)
        if pruned:
            c.set_move_knn(knn)
        run_sweep(c, method=method, to_convergence=True)
        c.optimize_clusters()
        return c

    full, pruned = sweep(False), sweep(True)
    assert pruned.n_clusters == full.n_clusters
    np.testing.assert_allclose(pruned.total_likelihood, full.total_likelihood, rtol=1e-9)
    hom, com = _hom_com(full.clusters, pruned.clusters)  # same partition (relabel ok)
    assert hom == 1.0 and com == 1.0


def test_knn_pruned_keeps_the_split_move():
    """best_move_knn must still be able to peel a cell into a fresh box even when
    no neighbour points there: an empty box is always in its candidate set."""
    from fastcellstates.model import _dm_kernels as k

    rng = np.random.default_rng(0)
    G = 60
    pa, pc = rng.dirichlet(np.ones(G)), rng.dirichlet(np.ones(G))
    a = np.array([rng.multinomial(400, pa) for _ in range(12)]).T  # (G, 12) cluster A
    outlier = rng.multinomial(400, pc)[:, None]  # 1 very different cell
    data = np.hstack([a, outlier])  # outlier appended to A
    labels = np.zeros(13, dtype=np.int32)
    clst = Cluster(data, 64.0, c=labels, max_clusters=4, n_cache=300)

    m = 12  # the outlier
    knn = np.tile(np.arange(12, dtype=np.int32), (13, 1))  # neighbours = only cluster-0 cells
    clst.set_move_knn(knn)
    c_knn, d_knn = k.best_move_knn(m, clst.cells, clst.state, clst.prior)
    c_full, d_full = clst.get_best_move(m)  # get_best_move = full scan

    assert d_full > 0 and clst.cluster_sizes[c_full] == 0  # full scan splits m off
    assert clst.cluster_sizes[c_knn] == 0  # pruned splits too
    np.testing.assert_allclose(d_knn, d_full, rtol=1e-9)


# --------------------------------------------------------------------------- #
# numba-internal consistency
# --------------------------------------------------------------------------- #


def _hom_com(a, b):
    def ent(x):
        _, c = np.unique(x, return_counts=True)
        p = c / c.sum()
        return -(p * np.log(p)).sum()

    def cond(x, y):
        h, n = 0.0, len(x)
        for yv in np.unique(y):
            s = x[y == yv]
            _, c = np.unique(s, return_counts=True)
            p = c / c.sum()
            h += (len(s) / n) * (-(p * np.log(p)).sum())
        return h

    Ha, Hb = ent(a), ent(b)
    return (1.0 if Ha == 0 else 1 - cond(a, b) / Ha, 1.0 if Hb == 0 else 1 - cond(b, a) / Hb)


def test_mcmc_recovers_truth(toy):
    data, labels = toy
    for seed in range(3):
        clst = Cluster(data, 256.0, n_cache=500, seed=seed + 1)
        clst.biased_monte_carlo_sampling(N_steps=clst.N_samples, tries_per_step=2000)
        clst.optimize_clusters()
        hom, _ = _hom_com(labels, clst.clusters)
        assert hom > 0.9, (seed, hom)
        tot, _ = partition_loglik(DATA_M, clst.clusters, clst.dirichlet_pseudocounts)
        np.testing.assert_allclose(clst.total_likelihood, tot, rtol=1e-6)


def test_incremental_likelihood_drift_bounded(toy):
    data, _ = toy
    clst = Cluster(data, 256.0, n_cache=500, seed=1)
    for _ in range(8):
        try:
            clst.biased_monte_carlo_sampling(N_steps=clst.N_samples, tries_per_step=1500)
        except RuntimeError:
            break
    like_running = clst.likelihood.copy()
    total_running = clst.total_likelihood
    clst.refresh_likelihood()
    np.testing.assert_allclose(like_running, clst.likelihood, rtol=0, atol=1e-4)
    np.testing.assert_allclose(total_running, clst.total_likelihood, rtol=1e-9)


def test_incremental_min_index_freezes_head(toy):
    data, labels = toy
    N = data.shape[1]
    frozen = N // 2
    init = np.concatenate([labels[:frozen], np.arange(3, 3 + N - frozen)])
    clst = Cluster(data, 256.0, c=init.copy(), n_cache=400, seed=1)
    head = clst.clusters[:frozen].copy()
    with contextlib.suppress(RuntimeError):
        clst.biased_monte_carlo_sampling(N_steps=40, tries_per_step=3000, min_index=frozen)
    np.testing.assert_array_equal(clst.clusters[:frozen], head)


def test_sparse_input_matches_dense_input(toy):
    import scipy.sparse as sp

    from fastcellstates.model import _dm_kernels as _nc
    from fastcellstates.moves import run_mcmc

    data, _ = toy
    variants = {
        "dense": data,
        "csc": sp.csc_matrix(data),
        "csr_transposed": sp.csr_matrix(data.T).T,
    }
    built = {n: Cluster(d, 256.0, n_cache=200, seed=3) for n, d in variants.items()}
    ref = built["dense"]
    for clst in built.values():
        np.testing.assert_allclose(clst.likelihood, ref.likelihood, rtol=1e-12)
        np.testing.assert_array_equal(clst.cluster_umi_counts, ref.cluster_umi_counts)
    assert built["csc"].data is None and built["csr_transposed"].data is None
    assert ref.data is not None
    np.testing.assert_array_equal(built["csc"].umi_data, ref.umi_data)

    def run_fresh(clst, seed):
        _nc.seed_rng(np.uint64(seed))
        np.random.seed(seed)
        run_mcmc(clst, N_steps=clst.N_samples, tries_per_step=2000, log_level="ERROR")
        return clst.clusters.copy()

    parts = {n: run_fresh(c, 11) for n, c in built.items()}
    for p in parts.values():
        np.testing.assert_array_equal(p, parts["dense"])


def test_sparse_input_gene_masking(toy):
    import scipy.sparse as sp

    data, _ = toy
    padded = np.vstack([data, np.zeros((1, data.shape[1]), dtype=data.dtype)])
    dense = Cluster(padded, None, n_cache=200, seed=1)
    sparse = Cluster(sp.csc_matrix(padded), None, n_cache=200, seed=1)
    assert dense.G == sparse.G < padded.shape[0]
    np.testing.assert_allclose(
        dense.dirichlet_pseudocounts, sparse.dirichlet_pseudocounts, rtol=1e-12
    )
    for alpha in (128.0, 512.0):
        dense.set_dirichlet_pseudocounts(alpha, n_cache=0)
        sparse.set_dirichlet_pseudocounts(alpha, n_cache=0)
        np.testing.assert_allclose(dense.likelihood, sparse.likelihood, rtol=1e-10)


def test_full_pipeline_runs(toy):
    from fastcellstates.analysis import get_hierarchy_df, marker_score_table
    from fastcellstates.moves import run_mcmc

    data, labels = toy
    clst = Cluster(data, 256.0, n_cache=500, seed=1)
    run_mcmc(clst, N_steps=clst.N_samples, tries_per_step=2000, log_level="ERROR")
    hierarchy, delta_hist = clst.get_cluster_hierarchy()
    hdf = get_hierarchy_df(hierarchy, delta_hist)
    scores = marker_score_table(clst, hdf)
    assert scores.shape == (hdf.shape[0], clst.G)
    assert not np.isnan(scores).any()
    hom, _ = _hom_com(labels, clst.clusters)
    assert hom > 0.9


def test_scipy_hierarchy_is_valid_linkage(toy):
    """get_scipy_hierarchy's Z must satisfy scipy's own linkage invariants
    (Z[:, 3] = size of the newly formed cluster, growing 2..n_leaves) so that
    scipy.cluster.hierarchy.dendrogram (-> plot_hierarchy_scipy) can consume
    it.  Regression for a copy-paste bug (clustersize initialised to each
    cluster's rank instead of 1, inherited from upstream) that scipy's
    dendrogram() catches as "excessive observations in a cluster" -- silent
    on some inputs, since nothing exercised this path until now."""
    from scipy.cluster.hierarchy import is_valid_linkage

    from fastcellstates.analysis import get_hierarchy_df, get_scipy_hierarchy

    data, _ = toy
    clst = Cluster(data, 256.0, n_cache=500, seed=1)
    hierarchy, delta_hist = clst.get_cluster_hierarchy()
    hdf = get_hierarchy_df(hierarchy, delta_hist)

    Z, leaf_labels = get_scipy_hierarchy(hdf, return_labels=True)
    assert is_valid_linkage(Z, throw=True)
    assert Z[-1, 3] == len(leaf_labels)  # the final merge holds every leaf
