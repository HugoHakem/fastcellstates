"""Block modules: partition (over-partition), model (Minka theta fit / posterior),
moves (coordinate ascent)."""

import numpy as np
from oracle import partition_loglik

from fastcellstates import partition
from fastcellstates.core import Cluster
from fastcellstates.graph import cell_knn
from fastcellstates.model import DirichletMultinomial
from fastcellstates.moves import coordinate_ascent

GOLD = np.load("test/data/cython_golden.npz")
DATA = GOLD["data"]
GMASK = DATA.sum(1) > 0
DATA_M = DATA[GMASK]


def test_singletons():
    lab = partition.singletons(50)
    assert lab.shape == (50,) and lab.dtype == np.int32
    assert (lab == np.arange(50)).all()


def test_over_partition_valid_labelling():
    knn = cell_knn(np.ascontiguousarray(DATA_M), k=15, metric="pca", n_pcs=20)
    for method, kw in (
        ("cpm", dict(resolution=0.2)),
        ("leiden_rbc", dict(resolution=5.0)),
        ("walktrap", dict(n_groups=25)),
    ):
        lab = partition.over_partition(knn, algorithm=method, seed=1, **kw)
        assert lab.shape == (DATA_M.shape[1],)
        K0 = int(lab.max()) + 1
        assert 1 <= K0 <= DATA_M.shape[1]
        assert set(np.unique(lab)) == set(range(K0))  # contiguous 0..K0-1


def _minka(C, lam, t0):
    return DirichletMultinomial(t0, lam).fit_theta(C).theta


def test_minka_theta_recovers_true_scale():
    # multi-cluster DM synthetic: 15 clusters ~ DM(theta_true * lambda), each
    # holding cells ~ DM within the cluster.  fit_theta should recover
    # ~theta_true from any starting point.
    rng = np.random.default_rng(0)
    G, K, cells_per, theta_true = 300, 15, 40, 800.0
    lam = rng.dirichlet(np.ones(G))
    fc = rng.dirichlet(theta_true * lam, size=K)
    C = np.zeros((K, G))
    for c in range(K):
        for _ in range(cells_per):
            C[c] += rng.multinomial(1500, rng.dirichlet(theta_true * fc[c]))
    vals = [_minka(C, lam, t0) for t0 in (50.0, 800.0, 40000.0, 2e5)]
    assert max(vals) / min(vals) < 1.01  # same fixed point
    assert 0.5 < np.mean(vals) / theta_true < 2.0  # right ballpark
    # fixed-point property
    ts = vals[0]
    assert abs(np.log(_minka(C, lam, ts)) - np.log(ts)) < 1e-4


def test_coordinate_ascent_converges():
    lab0 = np.repeat(np.arange(6), DATA_M.shape[1] // 6 + 1)[: DATA_M.shape[1]].astype(np.int32)

    def recluster(theta):
        K = int(lab0.max()) + 1
        c = Cluster(DATA_M, float(theta), c=lab0.copy(), max_clusters=K, n_cache=300)
        c.optimize_clusters()
        return c

    t_lo, c_lo = coordinate_ascent(recluster, 64.0)
    t_hi, _ = coordinate_ascent(recluster, 8192.0)
    assert abs(np.log(t_lo) - np.log(t_hi)) < 0.1  # same fixed point both ways
    # returned Cluster's LL matches the oracle at its Theta
    tot, _ = partition_loglik(DATA_M, c_lo.clusters, c_lo.dirichlet_pseudocounts)
    np.testing.assert_allclose(c_lo.total_likelihood, tot, rtol=1e-6)


def test_posterior_freq_normalised():
    C = np.array([[10.0, 0, 5], [0, 20, 1]])
    lam = np.array([0.5, 0.3, 0.2])
    dm = DirichletMultinomial(100.0, lam)
    for kind in ("mean", "mode"):
        f = dm.posterior_freq(C, kind=kind)
        np.testing.assert_allclose(f.sum(1), 1.0)
        assert (f >= 0).all()
