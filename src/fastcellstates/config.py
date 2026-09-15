"""
Run configuration.

``Config`` is a nested dataclass, one sub-config per block, and is the
single knob for both the Python API (``fastcellstates.run(data, cfg)``) and the CLI
(``--cfg.moves.sweep jacobi`` etc.).  ``PRESETS`` names the two endpoints:
``fast`` is the warm-start summariser (the dataclass defaults), ``exact``
reproduces the paper's from-singletons MCMC.
"""

from dataclasses import dataclass, field


@dataclass
class GraphCfg:
    """kNN graph over cells (feeds the search only, never the likelihood)."""

    n_pcs: int = 50
    k: int = 30
    """kNN for the over-partition (Leiden) graph."""
    metric: str = "pca"
    """graph.cell_knn distance: pca | cosine_log1p | sanity (experimental,
    see graph._sanity_delta)."""


@dataclass
class InitCfg:
    """How the starting partition is formed."""

    source: str = "over_partition"
    """``over_partition`` (community-detect a similarity graph) | ``singletons``."""
    algorithm: str = "cpm"
    """community algorithm for ``over_partition``: cpm | leiden_rbc | walktrap."""
    resolution: float = 0.1
    """cpm / leiden_rbc resolution (cpm 0.1 ~ N/40, size-stable)."""
    gamma: float = 35.0
    """walktrap: cut to ``round(N / gamma)`` groups."""


@dataclass
class MovesCfg:
    """The search over the partition and the concentration Theta."""

    mcmc: bool = False
    """run the paper's from-singletons Metropolis search (needs source=singletons)."""
    mcmc_tries: int = 1000
    """MCMC move proposals per step."""
    merge: bool = True
    """agglomerative DM merge: coarsen the over-partition to the DM optimum."""
    sweep: str = "jacobi"
    """cell-reassignment pass: jacobi | gauss_seidel | none."""
    sweep_to_convergence: bool = True
    sweep_prune_k: int = 0
    """0 = full candidate scan; >0 = restrict each cell's sweep moves to the
    clusters of its k nearest neighbours (near-lossless at 100; warm-start only)."""


@dataclass
class ModelCfg:
    """The generative model (supp. info §A1; see model.base.Model)."""

    kind: str = "dirichlet_multinomial"
    """the only model today."""
    theta: float = 0.0
    """Theta: 0 -> depth heuristic; >0 -> fixed value."""
    theta_method: str = "log_search"
    """log_search | coordinate_ascent | doubling | fixed."""
    theta_rounds: int = 10
    """coordinate_ascent: max (fit Theta <-> recluster) alternations, a hard
    cap not a target (each round reclusters from scratch, so nothing
    guarantees convergence by round ``theta_rounds``).  log_search: max
    Theta probes for Brent's method.  See ``moves.coordinate_ascent`` /
    ``moves.log_search``."""
    theta_tol: float = 0.02
    """coordinate_ascent: stop once ``|log(new_theta) - log(theta))| <
    theta_tol`` between rounds.  log_search: Brent's method ``xtol``, on
    log(Theta)."""
    n_cache: int = 10_000
    """target average lgamma-cache depth per gene (total budget n_genes *
    n_cache entries, water-filled per gene: low-expression genes get full
    coverage, the freed budget goes to the high-expression tail; see
    model._dm_kernels.build_prior)."""


@dataclass
class Config:
    """The full run configuration: one sub-config per block, plus top-level knobs."""

    graph: GraphCfg = field(default_factory=GraphCfg)
    init: InitCfg = field(default_factory=InitCfg)
    moves: MovesCfg = field(default_factory=MovesCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    seed: int = 1
    n_threads: int = 0
    """numba threads for the parallel Jacobi sweep. 0 = leave numba's default
    (every core it sees); >0 caps it (e.g. to be polite on a shared node).
    BLAS is pinned to <=16 around the PCA regardless; see graph._blas_limit."""
    transpose: bool = False
    """the loaded data is cells x genes (the opposite of this package's own
    genes x cells convention): transpose it right after loading, before
    anything else sees it.  ``.h5ad`` never needs this (obs/var already say
    which axis is which); it's for the shapeless formats (.tsv, .csv, .npy,
    .mtx) where a CLI run has no other way to say "my file is transposed"."""


PRESETS: dict[str, Config] = {
    "fast": Config(moves=MovesCfg(sweep_prune_k=100)),  # warm start -> kNN-pruned sweep
    "exact": Config(
        init=InitCfg(source="singletons"),
        moves=MovesCfg(mcmc=True, merge=True, sweep="gauss_seidel", sweep_to_convergence=False),
        model=ModelCfg(theta_method="doubling"),
    ),
}
