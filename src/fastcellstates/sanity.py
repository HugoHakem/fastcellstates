"""
Python-level wrapper around ``_sanity_kernels``: Breda et al.'s Sanity
correction and its uncertainty-aware cell-cell distance
(github.com/jmbreda/Sanity, Nat. Biotechnol. 2021), for use as the
``metric="sanity"`` option in ``graph.cell_knn`` -- see that module's
docstring for where this fits into the search, and ``_sanity_kernels`` for
the numerics this is ported from.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ._sanity_kernels import euclidean_distance_kernel, fit_all_genes, sanity_distance_kernel

V_METHODS = {"marg": 0, "mle": 1, "map": 2, "eap": 3}


@dataclass
class SanityFit:
    """Per-gene fit, in ``gene_mask``'s True order (all-zero genes dropped,
    matching ``Cluster``'s own convention elsewhere in this package)."""

    gene_mask: np.ndarray  # (G_in,) bool
    mu: np.ndarray  # (G_kept,)
    var_mu: np.ndarray  # (G_kept,)
    delta: np.ndarray  # (G_kept, N) posterior-mean log fold-change
    var_delta: np.ndarray  # (G_kept, N) posterior variance ("epsilon^2")
    var_gene: np.ndarray  # (G_kept,) fit prior variance v_g


def sanity_fit(data, vmin=0.001, vmax=50.0, numbin=160, v_method="map") -> SanityFit:
    """(G, N) counts -> per-gene/cell Sanity fit.  Defaults match the
    reference CLI's own (``-vmin``/``-vmax``/``-nbin``/``-v_m``)."""
    X = np.asarray(data.todense() if sp.issparse(data) else data, dtype=np.float64)
    gene_mask = X.sum(axis=1) > 0
    Xg = np.ascontiguousarray(X[gene_mask])
    N_c = Xg.sum(axis=0)
    N_c[N_c == 0] = 1.0

    deltav = np.log(vmax / vmin) / (numbin - 1)
    v_grid = vmin * np.exp(deltav * np.arange(numbin))

    vm = V_METHODS[v_method.lower()] if isinstance(v_method, str) else int(v_method)
    mu, var_mu, delta, var_delta, var_gene = fit_all_genes(Xg, N_c, v_grid, vm)
    return SanityFit(gene_mask, mu, var_mu, delta, var_delta, var_gene)


def sanity_distance(fit: SanityFit, s2n_cutoff=1.0, with_error_bar=True) -> np.ndarray:
    """(N, N) cell-cell distance, ``Sanity_distance``'s own defaults
    (signal/noise gene filter >= 1.0, uncertainty-weighted).  O(N^2 *
    G_kept): a compiled nested loop, same as the reference tool, and for
    the same reason -- this is a few-thousand-cells metric, not the
    ~50k-cell brute-force PCA path's league."""
    delta, var_delta, var_gene = fit.delta, fit.var_delta, fit.var_gene
    if s2n_cutoff > 0.0:
        mean_delta = delta.mean(axis=1, keepdims=True)
        var_across_cells = ((delta - mean_delta) ** 2).sum(axis=1) / (delta.shape[1] - 1)
        mean_eps2 = var_delta.mean(axis=1)
        keep = var_across_cells / np.maximum(mean_eps2, 1e-300) >= s2n_cutoff
        delta, var_delta, var_gene = delta[keep], var_delta[keep], var_gene[keep]

    if delta.shape[0] == 0:
        raise ValueError(
            f"no gene passed the signal/noise cutoff ({s2n_cutoff}); lower it or use more cells"
        )

    if not with_error_bar:
        return euclidean_distance_kernel(np.ascontiguousarray(delta.T))

    # de-shrink delta/epsilon2 before the distance calc: the fit's own
    # posterior mean already shrunk delta toward 0 by ~var_gene/(var_gene +
    # epsilon2); this factor undoes exactly that, since the distance's own
    # per-pair marginalization (below) needs the un-shrunk quantities to
    # apply its own (pair-specific) shrinkage correctly.
    denom = var_gene[:, None] - var_delta
    factor = np.where(
        denom > 0.0, var_gene[:, None] / np.maximum(denom, 1e-300), var_gene[:, None] / 1e-6
    )
    delta_r = np.ascontiguousarray((delta * factor).T)
    eps2_r = np.ascontiguousarray((var_delta * factor).T)
    return sanity_distance_kernel(delta_r, eps2_r, np.ascontiguousarray(var_gene))
