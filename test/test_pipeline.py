"""End-to-end pipeline: fast vs exact preset, Summary generative round-trips."""

import copy

import numpy as np
import pytest

import fastcellstates as fcs
from fastcellstates.summary import Summary


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(1)
    G, N, K, L, theta = 400, 300, 4, 1200, 300.0
    phi = rng.dirichlet(np.ones(G))
    fc = rng.dirichlet(theta * phi, size=K)
    labels = rng.integers(0, K, N)
    d = np.zeros((G, N), dtype=np.int64)
    for i in range(N):
        d[:, i] = rng.multinomial(L, fc[labels[i]])
    return d[d.sum(1) > 0], labels


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score

    return adjusted_rand_score(a, b)


def _fast_cfg():
    cfg = copy.deepcopy(fcs.PRESETS["fast"])
    cfg.init = fcs.InitCfg(source="over_partition", algorithm="leiden_rbc", resolution=3.0)
    cfg.graph = fcs.GraphCfg(k=15, n_pcs=20)
    cfg.model = fcs.ModelCfg(theta_method="fixed", theta=300.0)
    return cfg


def test_fast_preset_recovers_clusters(synthetic):
    d, truth = synthetic
    s = fcs.run(d, _fast_cfg())
    assert isinstance(s, Summary)
    assert s.n_states == 4
    assert _ari(truth, s.labels) > 0.95
    np.testing.assert_allclose(s.weights.sum(), 1.0)
    np.testing.assert_allclose(s.freq().sum(1), 1.0)


def test_exact_preset_recovers_clusters(synthetic):
    d, truth = synthetic
    cfg = copy.deepcopy(fcs.PRESETS["exact"])
    cfg.model = fcs.ModelCfg(theta_method="fixed", theta=300.0)  # skip the x2//2 probe
    s = fcs.run(d, cfg)
    assert _ari(truth, s.labels) > 0.95


def test_run_aligns_gene_names_to_kept_genes(synthetic):
    """Cluster drops all-zero-total genes; run() must filter the caller's
    ``genes`` the same way, so summ.genes lines up with summ.counts / summ.phi
    (regression: run() used to store the unfiltered names)."""
    d, _ = synthetic
    padded = np.vstack([d, np.zeros((7, d.shape[1]), dtype=d.dtype)])  # 7 never-expressed
    names = np.array([f"g{i}" for i in range(padded.shape[0])])
    s = fcs.run(padded, _fast_cfg(), genes=names)
    assert s.genes is not None
    assert s.genes.shape[0] == s.counts.shape[1] == s.phi.shape[0]
    assert not np.isin(s.genes, names[-7:]).any()  # the padded names were dropped
    np.testing.assert_array_equal(s.genes, names[: d.shape[0]])


def test_cfg_transpose_recovers_cells_by_genes_input(tmp_path, synthetic):
    """cfg.transpose lets a cells x genes file/array run correctly: the CLI
    has no other way to say "my file is transposed" (Python callers can just
    pass data.T themselves)."""
    import pandas as pd

    d, _ = synthetic
    cfg = copy.deepcopy(fcs.PRESETS["fast"])
    cfg.model = fcs.ModelCfg(theta_method="fixed", theta=300.0)
    s_normal = fcs.run(d, cfg)

    # raw-array path: pass the transposed array directly
    cfg_t = copy.deepcopy(cfg)
    cfg_t.transpose = True
    s_array = fcs.run(d.T, cfg_t)
    np.testing.assert_array_equal(s_normal.labels, s_array.labels)

    # file path: a cells x genes TSV, loaded through the CLI
    from fastcellstates import cli

    names = np.array([f"g{i}" for i in range(d.shape[0])])
    flipped = tmp_path / "flipped.tsv"
    pd.DataFrame(d.T, columns=names).to_csv(flipped, sep="\t")
    out = tmp_path / "out"
    cli.main(
        [
            str(flipped),
            "--cfg.transpose",
            "true",
            "--cfg.model.theta_method",
            "fixed",
            "--cfg.model.theta",
            "300",
            "-o",
            str(out),
        ]
    )
    s_file = Summary.load(out / "summary.npz")
    np.testing.assert_array_equal(s_normal.labels, s_file.labels)
    np.testing.assert_array_equal(s_file.genes, names)


def test_coordinate_ascent_moves_theta(synthetic):
    d, truth = synthetic
    cfg = _fast_cfg()
    cfg.model = fcs.ModelCfg(theta_method="coordinate_ascent")  # theta0 = heuristic
    s = fcs.run(d, cfg)
    assert _ari(truth, s.labels) > 0.9
    assert s.theta > 0


def test_run_phi_pins_prior_direction_independent_of_theta_method(synthetic):
    """phi overrides the prior's direction; theta_method still governs Theta on
    top of it, whether Theta is held fixed or searched."""
    d, _ = synthetic
    ext_phi = np.full(d.shape[0], 1.0 / d.shape[0])  # deliberately not the data's own phi

    cfg = _fast_cfg()  # theta_method="fixed", theta=300.0
    s = fcs.run(d, cfg, phi=ext_phi)
    np.testing.assert_allclose(s.phi, ext_phi)
    assert s.theta == 300.0

    cfg2 = _fast_cfg()
    cfg2.model = fcs.ModelCfg(theta_method="log_search")  # theta0 = heuristic, then searched
    s2 = fcs.run(d, cfg2, phi=ext_phi)
    np.testing.assert_allclose(s2.phi, ext_phi)  # phi stays pinned regardless


def test_summary_generative_roundtrip(synthetic):
    d, _ = synthetic
    s = fcs.run(d, _fast_cfg())

    xr = s.reconstruct(d, rng=0)
    assert xr.shape == d.shape
    # per-gene mean of the reconstruction tracks the input
    rel = np.abs(xr.mean(1) - d.mean(1)) / (d.mean(1) + 1e-6)
    assert np.median(rel) < 0.5

    new = s.sample(50, rng=1)
    assert new.shape[1] == 50 and new.dtype == np.int64

    pred = s.predict_state(d[:, :30])
    assert pred.shape == (30,) and set(np.unique(pred)) <= set(range(s.n_states))


def test_summary_save_load(tmp_path, synthetic):
    d, _ = synthetic
    s = fcs.run(d, _fast_cfg())
    p = tmp_path / "s.npz"
    s.save(p)
    r = Summary.load(p)
    np.testing.assert_array_equal(r.labels, s.labels)
    np.testing.assert_allclose(r.freq(), s.freq())
    assert r.theta == s.theta
    np.testing.assert_allclose(r.phi, s.phi)


def test_summary_hierarchy_is_lazy_and_deterministic(tmp_path, synthetic):
    d, _ = synthetic
    s = fcs.run(d, _fast_cfg())

    h = s.hierarchy()
    assert h.shape == (s.n_states - 1, 3)

    # a coarser cut is a valid partition and a strict coarsening of s.labels
    coarse = s.cut(2)
    assert coarse.shape == s.labels.shape
    assert len(np.unique(coarse)) == 2
    for c in np.unique(s.labels):  # each fine state maps to one coarse state
        assert len(np.unique(coarse[s.labels == c])) == 1

    m = s.markers()
    assert m.shape == (s.n_states - 1, s.counts.shape[1])
    assert not np.isnan(m).any()  # +-inf allowed, NaN not

    # save/load -> byte-identical hierarchy (pure function of counts + theta + phi)
    p = tmp_path / "s.npz"
    s.save(p)
    h2 = Summary.load(p).hierarchy()
    np.testing.assert_allclose(
        np.array(h["delta_LL"].tolist(), float), np.array(h2["delta_LL"].tolist(), float), rtol=1e-9
    )


def test_import_is_matplotlib_free():
    """`import fastcellstates` must not pull matplotlib: it's only in the `plot` extra."""
    import subprocess
    import sys

    code = (
        "import sys, fastcellstates; "
        "mpl = [m for m in sys.modules if m.split('.')[0] == 'matplotlib']; "
        "assert not mpl, mpl"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_run_and_config_roundtrip(tmp_path, capsys, synthetic):
    import pandas as pd

    from fastcellstates import cli

    d, _ = synthetic
    data = tmp_path / "counts.tsv"
    pd.DataFrame(
        d,
        index=[f"g{i}" for i in range(d.shape[0])],
        columns=[f"cell_{i}" for i in range(d.shape[1])],
    ).to_csv(data, sep="\t")
    args = [
        str(data),
        "--cfg.init.algorithm",
        "leiden_rbc",
        "--cfg.init.resolution",
        "3",
        "--cfg.model.theta_method",
        "fixed",
        "--cfg.model.theta",
        "300",
    ]
    cli.main([*args, "-o", str(tmp_path)])
    assert (tmp_path / "summary.npz").exists()
    n_states = Summary.load(tmp_path / "summary.npz").n_states
    capsys.readouterr()  # drop the run output

    # --print_config exits; capture the YAML, replay it, same result
    with pytest.raises(SystemExit):
        cli.main([*args, "--print_config"])
    yaml_text = capsys.readouterr().out
    cfg_file = tmp_path / "run.yaml"
    cfg_file.write_text(yaml_text)

    out2 = tmp_path / "replay"
    cli.main(["--config", str(cfg_file), "-o", str(out2)])
    assert Summary.load(out2 / "summary.npz").n_states == n_states
