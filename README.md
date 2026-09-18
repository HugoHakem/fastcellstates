<p align="center">
  <img src="docs/_static/fastcellstates.svg" alt="fastcellstates" width="1000">
</p>

<p align="center">
  <a href="https://github.com/HugoHakem/fastcellstates/actions/workflows/lint.yaml"><img src="https://github.com/HugoHakem/fastcellstates/actions/workflows/lint.yaml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/HugoHakem/fastcellstates/actions/workflows/test.yaml"><img src="https://github.com/HugoHakem/fastcellstates/actions/workflows/test.yaml/badge.svg" alt="Test"></a>
  <a href="https://github.com/HugoHakem/fastcellstates/actions/workflows/build.yaml"><img src="https://github.com/HugoHakem/fastcellstates/actions/workflows/build.yaml/badge.svg" alt="Check Build"></a>
</p>

`fastcellstates` is a numba reimplementation of [`cellstates`](https://github.com/nimwegenLab/cellstates).

[`docs/changes.md`](docs/changes.md) covers what differs and why.

As the original, it supports the analysis of UMI-based single-cell RNA-seq data:
it infers clusters of cells that are in the same gene expression state, where all
remaining within-cluster heterogeneity is explained by expected measurement
noise.

The notebooks in [`docs/notebooks`](docs/notebooks/README.md) will help you get started.

## This fork

- **Pure Python:**
  - The hot kernels are numba-JIT'd at runtime (disk-cached)
  - It reads `.h5ad` / `.mtx` without ever densifying
  - For files too large to load whole, `io.peek_h5ad` / `io.read_h5ad_subset` / `io.iter_h5ad_chunks` give chunked, backed-mode access instead (see [`docs/notebooks/large_datasets.ipynb`](docs/notebooks/large_datasets.ipynb))
- **Composable blocks:**
  - `io → graph → partition → model → moves → summary`, driven by `fastcellstates.run(data, cfg)` / the `fastcellstates` CLI.
- **A generative `Summary`:**
  - The output is a compact model (labels + per-state Dirichlet-multinomial frequency vectors + mixing weights + library-size model)
  - `.sample()`, `.reconstruct()`, and `.predict_state()` all work directly from it, without re-running anything

## Installation

With [pixi](https://pixi.sh) (`pixi install` solves the environment *and*
installs `fastcellstates` editable):

```sh
pixi install
pixi run test
```

Or with pip, from a local clone:

```sh
pip install -e .
```

Or directly from GitHub, no clone needed (not on PyPI yet):

```sh
pip install git+https://github.com/HugoHakem/fastcellstates.git
```

## Command line

```sh
fastcellstates data.h5ad --preset fast -o out/
```

`data` is a table of integer UMI counts:

- `.tsv`, `.txt`, `.csv`, `.npy`, or `.mtx` (rows genes, columns cells)
- `.h5ad` (`adata.X`, cells × genes; always handled correctly, since `obs`/`var` say which axis is which)
- If a `.tsv`/`.npy`/`.mtx` file happens to be cells × genes instead, pass `--cfg.transpose true`.
- Multiple files are concatenated column-wise (same genes assumed).

**Presets** (`--preset`):

| preset | what it does |
|---|---|
| `fast` (default) | PCA-kNN → Leiden over-partition → DM-optimal merge → kNN-pruned cell sweep, Theta fit by a Brent search (`log_search`). Seconds to a minute; a compact generative summary. |
| `exact` | the paper's from-singletons MCMC + merge + sweep, Theta fit by `doubling`. Slow; the reference. |

Every field is overridable as `--cfg.<block>.<field>`:

```sh
fastcellstates data.h5ad --preset fast --cfg.model.theta_method fixed --cfg.model.theta 4096
fastcellstates data.h5ad --cfg.init.algorithm walktrap --cfg.init.gamma 40
fastcellstates data.h5ad --print_config > run.yaml   # save the fully-resolved config
fastcellstates data.h5ad --config run.yaml           # ... and replay it exactly
fastcellstates --help                                # every field, with its default
```

<details>
<summary><code>fastcellstates --help</code></summary>

```text
usage: fastcellstates [-o OUT] [--preset {fast,exact}] [--config CONFIG]
                      [--cfg CONFIG] [--cfg.graph CONFIG]
                      [--cfg.graph.n_pcs N_PCS] [--cfg.graph.k K]
                      [--cfg.init CONFIG] [--cfg.init.source SOURCE]
                      [--cfg.init.algorithm ALGORITHM]
                      [--cfg.init.resolution RESOLUTION]
                      [--cfg.init.gamma GAMMA] [--cfg.moves CONFIG]
                      [--cfg.moves.mcmc {true,false}]
                      [--cfg.moves.mcmc_tries MCMC_TRIES]
                      [--cfg.moves.merge {true,false}]
                      [--cfg.moves.sweep SWEEP]
                      [--cfg.moves.sweep_to_convergence {true,false}]
                      [--cfg.moves.sweep_prune_k SWEEP_PRUNE_K]
                      [--cfg.model CONFIG] [--cfg.model.kind KIND]
                      [--cfg.model.theta THETA]
                      [--cfg.model.theta_method THETA_METHOD]
                      [--cfg.model.theta_rounds THETA_ROUNDS]
                      [--cfg.model.theta_tol THETA_TOL]
                      [--cfg.model.n_cache N_CACHE] [--cfg.seed SEED]
                      [--cfg.n_threads N_THREADS]
                      [--cfg.transpose {true,false}]
                      data [data ...]

Dirichlet-multinomial cell-state clustering.

positional arguments:
  data                  UMI file(s): .h5ad / .mtx / .tsv / .csv / .npy
                        (required)

options:
  -h, --help            Show this help message and exit.
  -o OUT, --out OUT     dir for summary.npz + labels.txt (default: null)
  --preset {fast,exact}
                        starting config (--cfg.* override) (default: fast)
  --config CONFIG       load a YAML config
  --print_config [=flags]
                        Print the configuration after applying all other
                        arguments and exit. The optional flags customizes the
                        output and are one or more keywords separated by
                        comma. The supported flags are: skip_default,
                        skip_unset.

The full run configuration: one sub-config per block, plus top-level knobs:
  --cfg CONFIG          Path to a configuration file.
  --cfg.seed SEED       (type: int, default: 1)
  --cfg.n_threads N_THREADS
                        numba threads for the parallel Jacobi sweep. 0 = leave
                        numba's default (every core it sees); >0 caps it (e.g.
                        to be polite on a shared node). BLAS is pinned to <=16
                        around the PCA regardless; see graph._blas_limit.
                        (type: int, default: 0)
  --cfg.transpose {true,false}
                        the loaded data is cells x genes (the opposite of this
                        package's own genes x cells convention): transpose it
                        right after loading, before anything else sees it.
                        ``.h5ad`` never needs this (obs/var already say which
                        axis is which); it's for the shapeless formats (.tsv,
                        .csv, .npy, .mtx) where a CLI run has no other way to
                        say "my file is transposed". (type: bool, default:
                        False)

PCA-kNN graph over cells (feeds the search only, never the likelihood):
  --cfg.graph CONFIG    Path to a configuration file.
  --cfg.graph.n_pcs N_PCS
                        (type: int, default: 50)
  --cfg.graph.k K       kNN for the over-partition (Leiden) graph. (type: int,
                        default: 30)

How the starting partition is formed:
  --cfg.init CONFIG     Path to a configuration file.
  --cfg.init.source SOURCE
                        ``over_partition`` (community-detect a similarity
                        graph) | ``singletons``. (type: str, default:
                        over_partition)
  --cfg.init.algorithm ALGORITHM
                        community algorithm for ``over_partition``: cpm |
                        leiden_rbc | walktrap. (type: str, default: cpm)
  --cfg.init.resolution RESOLUTION
                        cpm / leiden_rbc resolution (cpm 0.1 ~ N/40, size-
                        stable). (type: float, default: 0.1)
  --cfg.init.gamma GAMMA
                        walktrap: cut to ``round(N / gamma)`` groups. (type:
                        float, default: 35.0)

The search over the partition and the concentration Theta:
  --cfg.moves CONFIG    Path to a configuration file.
  --cfg.moves.mcmc {true,false}
                        run the paper's from-singletons Metropolis search
                        (needs source=singletons). (type: bool, default:
                        False)
  --cfg.moves.mcmc_tries MCMC_TRIES
                        MCMC move proposals per step. (type: int, default:
                        1000)
  --cfg.moves.merge {true,false}
                        agglomerative DM merge: coarsen the over-partition to
                        the DM optimum. (type: bool, default: True)
  --cfg.moves.sweep SWEEP
                        cell-reassignment pass: jacobi | gauss_seidel | none.
                        (type: str, default: jacobi)
  --cfg.moves.sweep_to_convergence {true,false}
                        (type: bool, default: True)
  --cfg.moves.sweep_prune_k SWEEP_PRUNE_K
                        0 = full candidate scan; >0 = restrict each cell's
                        sweep moves to the clusters of its k nearest
                        neighbours (near-lossless at 100; warm-start only).
                        (type: int, default: 100)

The generative model (supp. info §A1; see model.base.Model):
  --cfg.model CONFIG    Path to a configuration file.
  --cfg.model.kind KIND
                        the only model today. (type: str, default:
                        dirichlet_multinomial)
  --cfg.model.theta THETA
                        Theta: 0 -> depth heuristic; >0 -> fixed value. (type:
                        float, default: 0.0)
  --cfg.model.theta_method THETA_METHOD
                        log_search | coordinate_ascent | doubling | fixed.
                        (type: str, default: log_search)
  --cfg.model.theta_rounds THETA_ROUNDS
                        coordinate_ascent: max (fit Theta <-> recluster)
                        alternations, a hard cap not a target (each round
                        reclusters from scratch, so nothing guarantees
                        convergence by round ``theta_rounds``). log_search:
                        max Theta probes for Brent's method. See
                        ``moves.coordinate_ascent`` / ``moves.log_search``.
                        (type: int, default: 10)
  --cfg.model.theta_tol THETA_TOL
                        coordinate_ascent: stop once ``|log(new_theta) -
                        log(theta))| < theta_tol`` between rounds. log_search:
                        Brent's method ``xtol``, on log(Theta). (type: float,
                        default: 0.02)
  --cfg.model.n_cache N_CACHE
                        target average lgamma-cache depth per gene (total
                        budget n_genes * n_cache entries, water-filled per
                        gene: low-expression genes get full coverage, the
                        freed budget goes to the high-expression tail; see
                        model._dm_kernels.build_prior). (type: int, default:
                        10000)
```

</details>

**Output** (`-o out/`): `summary.npz` (load with `fastcellstates.Summary.load`) and
`labels.txt` (one cell-state id per cell).

## Python

```python
import fastcellstates as fcs

summ = fcs.run("data.h5ad")                 # -> Summary
summ.labels                                 # (N,) cell-state per cell
summ.freq()                                 # (K, G) per-state frequency vectors
summ.sample(1000)                           # (G, 1000) freshly drawn cells
summ.reconstruct(counts)                    # resample the input at its own depths
summ.predict_state(new_counts)              # place held-out cells
summ.hierarchy(); summ.cut(10); summ.markers()   # lazy, from the K count vectors
summ.save("out/summary.npz")
```

Pin the prior's direction (`phi`) to an externally estimated reference instead
of this run's own data, e.g. to keep several subsets directly comparable
against a shared control population -- Theta is untouched and still follows
`cfg.model.theta_method` as usual (see `docs/changes.md#an-externally-fixed-phi`):

```python
phi_ref = fcs.global_phi(control_counts)   # (G,) reference profile
summ = fcs.run(subset_counts, cfg, phi=phi_ref)
```

The low-level `Cluster` optimiser and `run_mcmc` are still exposed for custom
pipelines. See [`docs/notebooks`](docs/notebooks/README.md) for analysis and interpretation
examples.

Hierarchy dendrograms (`fastcellstates.plot_hierarchy_scipy` / `plot_hierarchy_ete3`) need the `plot` extra:

```sh
pip install 'fastcellstates[plot]'
```

The `ete3` renderer additionally needs conda-forge `ete3` (its PyPI build is broken).

## Testing

`pixi run test` (`pytest -q`) runs `test/`:

- `test_kernels.py`: the numba kernels vs a pure-numpy Dirichlet-multinomial
  reference (`fastcellstates.model._dm_reference`) and a frozen Cython golden file;
- `test_blocks.py`: `partition`, `model` (Minka θ fit, posteriors), `moves`;
- `test_pipeline.py`: end-to-end `fast`/`exact`, `Summary` round-trips.

`pixi run lint`: ruff + mypy.

## Citation

If you use `fastcellstates`, cite the method paper as below (and a reference to this reimplementation), see [`CITATION.cff`](CITATION.cff).

```bibtex
@article{grobeckerIdentifyingCellStates2024,
  title = {Identifying Cell States in Single-Cell {{RNA-seq}} Data at Statistically Maximal Resolution},
  author = {Grobecker, Pascal and Sakoparnig, Thomas and family=Nimwegen, given=Erik, prefix=van, useprefix=false},
  date = {2024-07-12},
  journaltitle = {PLOS Computational Biology},
  doi = {10.1371/journal.pcbi.1012224},
  url = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012224},
}
```

## License

MIT (see [`LICENSE`](LICENSE)): © 2021 Pascal Grobecker (original `cellstates`), © 2026 Hugo Hakem (this work).
