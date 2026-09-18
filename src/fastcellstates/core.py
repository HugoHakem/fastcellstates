"""
``Cluster``: the Dirichlet-multinomial partition primitive.

``Cluster`` is the mutable partition state of ``model.DirichletMultinomial``,
whose generative assumptions (the DM marginal, the prior, Theta, phi) live
there.  It holds three namedtuples the ``_dm_kernels`` operate on:

    prior  : the Dirichlet prior + lgamma cache    (``_k.Prior``)
    cells  : the per-cell CSC of raw counts         (``_k.Cells``; immutable)
    state  : labels + per-box aggregates + move-kNN (``_k.State``; mutated)

and exposes the historical scalar/array API as properties.  The search
strategies (MCMC, sweeps, merge, Theta) are separate modules that take a
``Cluster`` and mutate it.
"""

from typing import cast

import numpy as np
import scipy.sparse as sp

from ._types import Counts, Sparse
from .model import _dm_kernels as _k


class Cluster:
    """Partition of cells under the Dirichlet-multinomial marginal likelihood.

    Parameters
    ----------
    d : (G, N) ndarray or scipy.sparse, genes x cells UMI counts.  Sparse
        input is kept sparse (no dense (G, N) array is ever materialised); a
        dense array is kept as-is (for ``umi_data``).  Note: this is the
        opposite of AnnData's own convention (``adata.X`` is cells x genes):
        pass ``adata.X.T``, or go through ``io.read("file.h5ad")``, which
        transposes for you.
    theta : float | None
        The Dirichlet concentration Theta (supp. info §A1 slot 4).  ``None``
        uses the depth heuristic ``2 ** round(log2(mean UMI))``.  Mutually
        exclusive with ``pseudocounts``.
    phi : (G,) ndarray | None
        Override the fixed profile (slot 5, eq. 12) that would otherwise be
        estimated from ``d`` itself (``gene_totals / grand_total``); must
        sum to 1.  Only meaningful alongside ``theta`` (not
        ``pseudocounts``): Theta still scales it and still gets fit/searched
        by ``moves`` exactly as usual, but the *direction* of the prior is
        pinned to ``phi`` regardless of what ``d`` looks like -- e.g. fit
        ``phi`` once on a reference population (``model.phi.global_phi``)
        and reuse it, unchanged, when clustering a different, smaller
        population you want directly comparable to it. A gene with
        ``phi_g == 0`` is dropped the same way a gene with zero local mass
        would be; a gene this subset never itself expresses but ``phi``
        gives nonzero mass to is *not* dropped (the local-``phi`` default
        only ever sees this subset's own zeros).
    pseudocounts : (G,) ndarray | None
        The full Dirichlet prior vector directly, bypassing the Theta*phi
        decomposition entirely (the DM marginal likelihood itself never
        assumes that shape; only ``moves``' Theta search does). Mutually
        exclusive with ``theta``/``phi``. This is the fully general escape
        hatch: a Theta later read back off the ``Cluster`` (``.theta``) is
        just ``pseudocounts.sum()``, whether or not that sum was chosen
        deliberately.
    c : (N,) int | None, initial labels (default: every cell its own cluster).
    genes : (G,), optional gene names.
    max_clusters : int, n_boxes; 0 -> N (one box per cell).
    n_cache : int, target average log-gamma cache depth per gene (total budget
        n_genes * n_cache entries, water-filled per gene; see
        model._dm_kernels.build_prior).
    seed : int
    """

    def __init__(
        self,
        d: Counts,
        theta=None,
        phi=None,
        pseudocounts=None,
        c=None,
        genes=None,
        max_clusters=0,
        n_cache=10_000,
        seed=1,
    ):
        input_is_sparse = sp.issparse(d)
        if sp.issparse(d):
            # cast: .astype() is defined on scipy's private internal base
            # class; newer stubs don't rebind its `Self` through a Union
            # receiver, so pyright infers the base class instead
            d = cast(Sparse, sp.csc_matrix(d).astype(np.int64, copy=False))
            gene_totals = np.asarray(d.sum(axis=1), dtype=np.float64).ravel()
        else:
            d = np.ascontiguousarray(d)
            gene_totals = d.sum(axis=1).astype(np.float64)
        G0, N0 = np.asarray(d.shape)  # csc .shape is typed Optional upstream; unwrap via numpy
        grand_total = float(gene_totals.sum())

        # ---- Dirichlet pseudo-counts / non-expressed gene filtering ----
        if pseudocounts is not None:
            if theta is not None or phi is not None:
                raise ValueError("pass either pseudocounts directly, or theta/phi, not both")
            pseudocounts = np.asarray(pseudocounts, dtype=np.float64)
            if pseudocounts.shape[0] != G0:
                raise ValueError("The shapes of the data and pseudocounts do not match")
            if np.any(pseudocounts <= 0):
                raise ValueError("all dirichlet pseudo-counts must be >0")
            gene_mask = None
        else:
            if theta is None:
                theta = 2.0 ** (np.round(np.log2(grand_total / N0)))
            else:
                theta = float(theta)
                if theta <= 0.0:
                    raise ValueError("dirichlet prior parameter must be > 0")
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                if phi.shape[0] != G0:
                    raise ValueError("phi has a different length than the data's genes")
                if not np.isclose(phi.sum(), 1.0, atol=1e-6):
                    # Prior.theta is pseudocounts.sum(), so an unnormalised phi would
                    # silently make the *actual* concentration theta*phi.sum(), not
                    # the theta the caller passed -- fail loud, don't guess a fix.
                    raise ValueError(f"phi must sum to 1 (got {phi.sum():.6g})")
                if np.any(phi < 0.0):
                    raise ValueError("phi must be non-negative")
                pseudocounts = theta * phi
            else:
                pseudocounts = theta * gene_totals / grand_total
            gene_mask = pseudocounts > 0
            if np.any(gene_mask):
                pseudocounts = pseudocounts[gene_mask]
                gene_totals = gene_totals[gene_mask]
                if sp.issparse(d):
                    d = sp.csc_matrix(d)[gene_mask, :]
                else:
                    d = np.ascontiguousarray(d[gene_mask, :])
            else:
                gene_mask = None

        self.G = int(pseudocounts.shape[0])
        self.N_samples = int(N0)
        n_boxes = max_clusters if max_clusters > 0 else self.N_samples

        if c is None:
            c = np.arange(self.N_samples, dtype=np.int32)
        elif self.N_samples != len(c):
            raise ValueError("the shapes of the data and clusters do not match")
        else:
            c = np.asarray(c)
            if np.max(c) >= n_boxes:
                raise ValueError("all cluster labels must be smaller than max_clusters")
            if np.min(c) < 0:
                raise ValueError("all cluster labels must be positive")
        # always copy: the move/merge kernels mutate ``state.labels`` in place,
        # and ``ascontiguousarray`` is a no-op (aliases the caller's array) when
        # ``c`` is already int32 and contiguous, e.g. a Leiden partition reused
        # across Theta-search rounds would otherwise be silently corrupted.
        labels = np.array(c, dtype=np.int32, copy=True)

        # per-cell CSC (cells = columns).  Sparse-origin input never touches a
        # dense (G, N) array; dense-origin input keeps it for umi_data.
        self.cells = _k.build_cells(d)
        self.data: np.ndarray | None = (
            None if input_is_sparse else np.ascontiguousarray(d, dtype=np.int64)
        )

        if genes is not None:
            genes = np.asarray(genes)
            if genes.shape[0] != G0:
                raise ValueError(
                    f"genes has length {genes.shape[0]}, but the data has {G0} rows (genes); "
                    "check the data is genes x cells, not transposed"
                )
            self.genes = genes[gene_mask] if gene_mask is not None else genes
        else:
            self.genes = None

        _k.seed_rng(np.uint64(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)  # np.random.permutation in the sweeps

        self._check_count_dtype()
        self.prior = _k.build_prior(pseudocounts, gene_totals, n_cache)

        sizes, gene_counts, box_umi_sum = _k.init_counts(self.cells, labels, n_boxes, self.G)
        move_knn = np.empty((self.N_samples, 0), dtype=np.int32)
        self.state = _k.State(labels, sizes, np.zeros(n_boxes), gene_counts, box_umi_sum, move_knn)
        self.state = self.state._replace(loglik=_k.init_likelihood(self.state, self.prior))

    # ------------------------------------------------------------------ #
    # (re)initialisation helpers
    # ------------------------------------------------------------------ #

    def _gene_totals(self):
        """Per-gene UMI total over all cells, from the CSC arrays (O(nnz))."""
        return np.bincount(self.cells.gidx, weights=self.cells.gval, minlength=self.G).astype(
            np.float64
        )

    def _check_count_dtype(self):
        """gene_counts is int32; a single cluster holding every cell must not
        overflow it, so bound the dataset-wide per-gene total."""
        gmax = float(self._gene_totals().max()) if self.G else 0.0
        if gmax >= 2**31 - 1:
            raise ValueError(
                f"a gene's dataset-wide UMI total ({gmax:.3g}) does not fit in "
                "int32; run incrementally or widen gene_counts to int64"
            )

    def _init_counts(self):
        n_boxes = self.state.loglik.shape[0]
        sizes, gene_counts, box_umi_sum = _k.init_counts(
            self.cells, self.state.labels, n_boxes, self.G
        )
        self.state = self.state._replace(
            sizes=sizes, gene_counts=gene_counts, box_umi_sum=box_umi_sum
        )

    def _init_likelihood(self):
        self.state = self.state._replace(loglik=_k.init_likelihood(self.state, self.prior))

    def refresh_likelihood(self):
        """Full recompute of the per-box likelihood array.  The move kernels
        maintain it incrementally (~1 ULP x n_moves drift); call this at every
        breakpoint so convergence decisions see exact values."""
        self._init_likelihood()

    def set_move_knn(self, knn_idx):
        """Restrict the deterministic cell sweeps to each cell's kNN clusters
        (plus the split move).

        ``knn_idx`` : (N_samples, k) int, neighbour cell indices (from
        ``graph.cell_knn``).  Near-lossless at k>=100 on a warm-started
        partition (misses only jumps to non-neighbour occupied clusters,
        ~0.1 % of moves); a from-singletons search still needs the full scan.
        Pass ``None`` to restore the full scan.
        """
        if knn_idx is None:
            mk = np.empty((self.N_samples, 0), dtype=np.int32)
        else:
            mk = np.ascontiguousarray(knn_idx, dtype=np.int32)
            if mk.ndim != 2 or mk.shape[0] != self.N_samples:
                raise ValueError("knn_idx must be (N_samples, k)")
        self.state = self.state._replace(move_knn=mk)

    # ------------------------------------------------------------------ #
    # MCMC
    # ------------------------------------------------------------------ #

    _MC_CHUNK = 5_000_000

    def biased_monte_carlo_sampling(self, N_steps=1, tries_per_step=1000, min_index=0, N_batch=0):
        if N_batch > 0:
            tries_per_step = 100 * N_batch * N_steps
        max_tries = N_steps * tries_per_step
        total_tries = total_successes = 0
        while total_tries < max_tries and total_successes < N_steps:
            budget = min(self._MC_CHUNK, max_tries - total_tries)
            target = N_steps - total_successes
            t, s = _k.biased_mc_moves_chunk(
                budget, target, min_index, self.cells, self.state, self.prior
            )
            total_tries += t
            total_successes += s
            if t == 0 and s == 0:
                break
        self.refresh_likelihood()  # reset incremental drift at the breakpoint
        if total_successes < N_steps:
            raise RuntimeError(
                f"Only {total_successes} moves found within loop limit. "
                "Consider raising tries_per_step"
            )
        return total_tries

    # ------------------------------------------------------------------ #
    # reconfiguration
    # ------------------------------------------------------------------ #

    def set_N_boxes(self, Nb_new):
        if Nb_new < self.n_clusters:
            raise ValueError("Nb_new must be >= the number of clusters")
        unique_sorted = np.sort(np.unique(self.state.labels))
        mapping = {i: int(c) for i, c in enumerate(unique_sorted)}

        new_labels = np.zeros(self.N_samples, dtype=np.int32)
        new_loglik = np.zeros(int(Nb_new), dtype=np.float64)
        old = self.state.labels
        for i in range(len(unique_sorted)):
            c = mapping[i]
            new_loglik[i] = self.state.loglik[c]
            new_labels[old == c] = i
        self.state = self.state._replace(labels=np.ascontiguousarray(new_labels), loglik=new_loglik)
        self._init_counts()
        return mapping

    def set_dirichlet_pseudocounts(self, theta=None, phi=None, pseudocounts=None, n_cache=-1):
        """Rebuild the prior at a new ``theta``/``phi``/``pseudocounts``, keeping
        the current partition.  Same three-way contract as the constructor
        (see its docstring); note this never remembers a ``phi`` the
        ``Cluster`` may have been built with -- pass it again explicitly to
        keep it pinned across the change.
        """
        gene_totals = self._gene_totals()
        if pseudocounts is not None:
            if theta is not None or phi is not None:
                raise ValueError("pass either pseudocounts directly, or theta/phi, not both")
            pseudocounts = np.asarray(pseudocounts, dtype=np.float64)
            if pseudocounts.shape[0] != self.G:
                raise ValueError("The shapes of the data and pseudocounts do not match")
            if np.any(pseudocounts <= 0):
                raise ValueError("all dirichlet pseudo-counts must be >0")
        else:
            if theta is None:
                raise ValueError("pass theta (optionally with phi), or pseudocounts")
            theta = float(theta)
            if theta <= 0.0:
                raise ValueError("dirichlet prior parameter must be > 0")
            if phi is not None:
                phi = np.asarray(phi, dtype=np.float64)
                if phi.shape[0] != self.G:
                    raise ValueError("phi has a different length than the data's genes")
                if not np.isclose(phi.sum(), 1.0, atol=1e-6):
                    raise ValueError(f"phi must sum to 1 (got {phi.sum():.6g})")
                if np.any(phi < 0.0):
                    raise ValueError("phi must be non-negative")
                pseudocounts = theta * phi
            else:
                pseudocounts = theta * gene_totals / gene_totals.sum()

        nc = self.prior.n_cache if n_cache <= 0 else int(n_cache)
        self.prior = _k.build_prior(pseudocounts, gene_totals, nc)
        self._init_likelihood()

    def set_clusters(self, new_clusters, max_clusters=0):
        new_clusters = np.asarray(new_clusters)
        if np.min(new_clusters) < 0:
            raise ValueError("all cluster labels must be positive")
        max_label = int(np.max(new_clusters))
        if max_clusters > max_label:
            self.set_N_boxes(max_clusters)
        elif max_label >= self.N_boxes:
            self.set_N_boxes(max_label + 1)
        self.state = self.state._replace(labels=np.ascontiguousarray(new_clusters, dtype=np.int32))
        self._init_counts()
        self._init_likelihood()

    # ------------------------------------------------------------------ #
    # merging / hierarchy
    # ------------------------------------------------------------------ #

    def _merge_hierarchical(self, LL_threshold, n_cluster_threshold):
        mh, dh = _k.merge_clusters_hierarchical(
            float(LL_threshold), int(n_cluster_threshold), self.state, self.prior
        )
        return [(int(a), int(b)) for a, b in mh], list(dh)

    def merge_clusters(self, LL_threshold=0.0, n_cluster_threshold=1):
        if n_cluster_threshold < 1:
            raise ValueError("n_cluster_threshold must be > 0")
        return self._merge_hierarchical(LL_threshold, n_cluster_threshold)

    def get_cluster_hierarchy(self):
        c = self.clusters.copy()
        hierarchy, delta_history = self._merge_hierarchical(-np.inf, 1)
        self.set_clusters(c)
        return hierarchy, delta_history

    def _merge_clusters_optimally(self):
        hierarchy, delta_history = self.get_cluster_hierarchy()
        if not delta_history:
            return
        total_delta = np.cumsum(delta_history)
        a_max = int(np.argmax(total_delta))
        if total_delta[a_max] > 0.0:
            for c1, c2 in hierarchy[: a_max + 1]:
                self.combine_two_clusters(c1, c2)

    def combine_two_clusters(self, c1, c2):
        delta = _k.find_cluster_distance(int(c1), int(c2), self.state, self.prior)
        _k.merge_two_clusters(int(c1), int(c2), delta, self.state)

    # ------------------------------------------------------------------ #
    # deterministic optimisation
    # ------------------------------------------------------------------ #

    def get_best_move(self, m, move_to=None):
        if move_to is not None:
            raise NotImplementedError("move_to subset not supported")
        c_best, best_delta = _k.best_move_full(int(m), self.cells, self.state, self.prior)
        return int(c_best), float(best_delta)

    def optimal_move(self, m, move_to=None):
        c_best, delta_LL = self.get_best_move(m, move_to)
        if delta_LL > 0.0:
            self.move_cell(m, c_best)
        else:
            c_best = int(self.state.labels[m])
        return c_best

    def move_cell(self, m, c_new):
        m, c_new = int(m), int(c_new)
        c_old = int(self.state.labels[m])
        if c_old == c_new:
            return
        d_old = _k.delta_LL_old(m, c_old, self.cells, self.state, self.prior)
        d_new = _k.delta_LL_new(m, c_new, self.cells, self.state, self.prior)
        _k.apply_move(m, c_new, d_old, d_new, self.cells, self.state)

    def optimize_clusters(self, merge_clusters=True, optimize_cells=True, set_N_boxes=True):
        if merge_clusters:
            self._merge_clusters_optimally()
        if set_N_boxes:
            self.set_N_boxes(self.n_clusters + 2)
        if optimize_cells:
            cell_iter = np.random.permutation(self.N_samples).astype(np.int64)
            _k.optimize_cells(cell_iter, self.cells, self.state, self.prior)
            self.refresh_likelihood()  # clear incremental drift

    # ------------------------------------------------------------------ #
    # expression state
    # ------------------------------------------------------------------ #

    def get_expressionstate(self, c):
        if self.state.sizes[c] == 0:
            raise ValueError(f"{c} is an empty cluster")
        f = self.cluster_umi_counts[:, c] + np.asarray(self.prior.pseudocounts) - 1.0
        f[f < 0.0] = 0.0
        return f / np.sum(f)

    def get_expressionstate_mv(self, c):
        if self.state.sizes[c] == 0:
            raise ValueError(f"{c} is an empty cluster")
        n_gc = self.cluster_umi_counts[:, c] + np.asarray(self.prior.pseudocounts)
        n_c = np.sum(n_gc)
        f = n_gc / n_c
        return f, (f * (1 - f)) / (n_c + 1.0)

    # ------------------------------------------------------------------ #
    # properties  (historical scalar/array API)
    # ------------------------------------------------------------------ #

    @property
    def N_boxes(self):
        return int(self.state.sizes.shape[0])

    @property
    def n_cache(self):
        return self.prior.n_cache

    @property
    def theta(self):
        """The Dirichlet concentration Theta -- ``dirichlet_pseudocounts.sum()``."""
        return self.prior.theta

    @property
    def phi(self):
        """(G,) the fixed profile this partition's prior currently points at,
        ``dirichlet_pseudocounts / theta`` -- whatever ``phi``/``pseudocounts``
        it was built or last set with."""
        return np.asarray(self.prior.pseudocounts, dtype=np.float64) / self.prior.theta

    @property
    def B(self):
        return self.prior.B

    @property
    def n_clusters(self):
        return int(np.count_nonzero(self.state.sizes))

    @property
    def total_likelihood(self):
        return float(np.sum(self.state.loglik))

    @property
    def likelihood(self):
        return np.asarray(self.state.loglik, dtype=np.float64)

    @property
    def clusters(self):
        return np.asarray(self.state.labels, dtype=np.int32)

    @property
    def cluster_sizes(self):
        return np.asarray(self.state.sizes, dtype=int)

    @property
    def cluster_umi_counts(self):
        """(G, n_boxes) total UMI per gene per cluster.  Internal storage is
        (n_boxes, G) for cache locality; this is a transposed view."""
        return np.asarray(self.state.gene_counts.T, dtype=int)

    @property
    def cluster_umi_sum(self):
        return np.asarray(self.state.box_umi_sum, dtype=int)

    @property
    def dirichlet_pseudocounts(self):
        return np.asarray(self.prior.pseudocounts, dtype=float)

    @property
    def model(self):
        """The ``DirichletMultinomial`` (Theta, phi) this partition optimises under.
        ``phi`` is over the kept genes, so ``model.pseudocounts ==
        dirichlet_pseudocounts``."""
        from .model import DirichletMultinomial

        return DirichletMultinomial(self.theta, self.phi)

    @property
    def umi_data(self):
        """Dense (G, N) UMI counts.  For sparse-origin input this materialises a
        full dense array on first access (can be large)."""
        if self.data is None:
            self.data = np.zeros((self.G, self.N_samples), dtype=np.int64)
            for m in range(self.N_samples):
                lo, hi = self.cells.ptr[m], self.cells.ptr[m + 1]
                self.data[self.cells.gidx[lo:hi], m] = self.cells.gval[lo:hi]
        return np.asarray(self.data, dtype=int)
