"""
fastcellstates: Dirichlet-multinomial clustering of scRNA-seq cells into
gene-expression states (supp. info §A1), with a Leiden warm start.

    import fastcellstates as fcs
    summ = fcs.run("data.h5ad")   # -> Summary: labels, per-state frequencies,
                                   #    .sample() / .reconstruct() / .predict_state()

``run(data, cfg)`` is the pipeline entry point (``Config`` / ``PRESETS`` pick
``fast`` vs ``exact``); ``Cluster`` is the lower-level partition primitive for
custom searches.  Hierarchy dendrograms need the ``plot`` extra and load
lazily; see below.
"""

from .analysis import (
    clusters_from_hierarchy,
    gene_contribution_table,
    get_cluster_distances,
    get_hierarchy_df,
    get_scipy_hierarchy,
    hierarchy_to_newick,
    marker_score_table,
    marker_scores,
)
from .config import PRESETS, Config, GraphCfg, InitCfg, ModelCfg, MovesCfg
from .core import Cluster
from .model import DirichletMultinomial, Model
from .moves import run_mcmc
from .pipeline import run
from .summary import Summary

__all__ = [
    "PRESETS",
    "Cluster",
    "Config",
    "DirichletMultinomial",
    "GraphCfg",
    "InitCfg",
    "Model",
    "ModelCfg",
    "MovesCfg",
    "Summary",
    "clusters_from_hierarchy",
    "gene_contribution_table",
    "get_cluster_distances",
    "get_hierarchy_df",
    "get_scipy_hierarchy",
    "hierarchy_to_newick",
    "marker_score_table",
    "marker_scores",
    "run",
    "run_mcmc",
]

# Plotting is optional (matplotlib; ete3 for the ete3 renderer).  Loaded lazily
# so `import fastcellstates` works without the `plot` extra; explicit access
# (`fastcellstates.plot_hierarchy_scipy`) triggers the import.  Deliberately not in
# `__all__`: `from fastcellstates import *` must not require matplotlib.
_LAZY = {"plot_hierarchy_scipy", "plot_hierarchy_ete3"}


def __getattr__(name):
    if name in _LAZY:
        try:
            from . import plotting
        except ImportError as e:
            raise AttributeError(
                f"fastcellstates.{name} needs the plot extra: pip install 'fastcellstates[plot]'"
            ) from e
        try:
            return getattr(plotting, name)
        except AttributeError:
            raise AttributeError(
                f"fastcellstates.{name} needs ete3 (conda-forge): the PyPI build is broken"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [*__all__, *_LAZY]
