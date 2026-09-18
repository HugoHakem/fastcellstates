"""Block modules: partition (over-partition), model (Minka theta fit / posterior),
moves (coordinate ascent)."""

import numpy as np
import pytest
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


def _minka(C, phi, t0):
    return DirichletMultinomial(t0, phi).fit_theta(C).theta


def test_minka_theta_recovers_true_scale():
    # multi-cluster DM synthetic: 15 clusters ~ DM(theta_true * phi), each
    # holding cells ~ DM within the cluster.  fit_theta should recover
    # ~theta_true from any starting point.
    rng = np.random.default_rng(0)
    G, K, cells_per, theta_true = 300, 15, 40, 800.0
    phi = rng.dirichlet(np.ones(G))
    fc = rng.dirichlet(theta_true * phi, size=K)
    C = np.zeros((K, G))
    for c in range(K):
        for _ in range(cells_per):
            C[c] += rng.multinomial(1500, rng.dirichlet(theta_true * fc[c]))
    vals = [_minka(C, phi, t0) for t0 in (50.0, 800.0, 40000.0, 2e5)]
    assert max(vals) / min(vals) < 1.01  # same fixed point
    assert 0.5 < np.mean(vals) / theta_true < 2.0  # right ballpark
    # fixed-point property
    ts = vals[0]
    assert abs(np.log(_minka(C, phi, ts)) - np.log(ts)) < 1e-4


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
    phi = np.array([0.5, 0.3, 0.2])
    dm = DirichletMultinomial(100.0, phi)
    for kind in ("mean", "mode"):
        f = dm.posterior_freq(C, kind=kind)
        np.testing.assert_allclose(f.sum(1), 1.0)
        assert (f >= 0).all()


def test_cluster_phi_overrides_local_direction():
    """phi pins the prior's direction; theta still scales it as usual."""
    theta = 500.0
    local_phi = DATA_M.sum(1) / DATA_M.sum()
    c_local = Cluster(DATA_M, theta, n_cache=300)
    np.testing.assert_allclose(c_local.dirichlet_pseudocounts, theta * local_phi)

    ext_phi = np.full(DATA_M.shape[0], 1.0 / DATA_M.shape[0])  # deliberately not local_phi
    c_ext = Cluster(DATA_M, theta, phi=ext_phi, n_cache=300)
    np.testing.assert_allclose(c_ext.dirichlet_pseudocounts, theta * ext_phi)
    assert not np.allclose(c_ext.dirichlet_pseudocounts, c_local.dirichlet_pseudocounts)


def test_cluster_phi_keeps_genes_local_totals_would_drop():
    """A gene this subset never itself expresses but phi gives mass to is kept,
    not dropped -- the opposite of the local-phi default's own-zeros filter."""
    d = DATA_M.copy()
    d[0, :] = 0  # gene 0 now has zero local mass
    theta = 500.0

    c_local = Cluster(d, theta, n_cache=300)
    assert d.shape[0] - 1 == c_local.G  # gene 0 dropped

    ext_phi = np.full(d.shape[0], 1.0 / d.shape[0])
    c_ext = Cluster(d, theta, phi=ext_phi, n_cache=300)
    assert d.shape[0] == c_ext.G  # gene 0 kept: phi says it has mass


def test_cluster_pseudocounts_matches_equivalent_theta_phi():
    """pseudocounts is the fully general escape hatch l used to be: passing
    theta*phi directly must build the exact same prior as passing theta/phi
    separately -- no behavior lost by splitting the old overloaded parameter."""
    theta = 500.0
    phi = np.full(DATA_M.shape[0], 1.0 / DATA_M.shape[0])
    c_split = Cluster(DATA_M, theta, phi=phi, n_cache=300)
    c_combined = Cluster(DATA_M, pseudocounts=theta * phi, n_cache=300)
    np.testing.assert_allclose(c_combined.dirichlet_pseudocounts, c_split.dirichlet_pseudocounts)
    assert c_combined.theta == pytest.approx(c_split.theta)


def test_cluster_pseudocounts_rejects_theta_or_phi_together():
    with pytest.raises(ValueError, match="not both"):
        Cluster(DATA_M, theta=1.0, pseudocounts=np.ones(DATA_M.shape[0]))
    with pytest.raises(ValueError, match="not both"):
        Cluster(
            DATA_M,
            phi=np.ones(DATA_M.shape[0]) / DATA_M.shape[0],
            pseudocounts=np.ones(DATA_M.shape[0]),
        )


def test_cluster_phi_rejects_mismatched_shape():
    with pytest.raises(ValueError, match="length"):
        Cluster(DATA_M, 500.0, phi=np.ones(DATA_M.shape[0] + 1))


def test_cluster_phi_rejects_unnormalised_input():
    """phi.sum() must be 1: Prior.theta = pseudocounts.sum(), so an unnormalised
    phi would silently detach the actual concentration from the theta passed in."""
    not_normalised = np.full(DATA_M.shape[0], 2.0 / DATA_M.shape[0])  # sums to 2
    with pytest.raises(ValueError, match="sum to 1"):
        Cluster(DATA_M, 500.0, phi=not_normalised)
