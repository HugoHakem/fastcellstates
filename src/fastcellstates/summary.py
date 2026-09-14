"""
``Summary``: the compact generative model that is the point of the pipeline.

A partition of the input cells into K states, each state a Dirichlet-multinomial
posterior over gene frequencies.  Sample new cells, reconstruct the input, or
place held-out cells, all without re-running anything.

    alpha_c ~ Dirichlet(Theta*lambda + C_c)                   (posterior, eq. 18)
    cell in state c, library size L   ~   Multinomial(L, alpha_c)

a fresh alpha per sampled cell (``Summary.sample``'s default): the proper
posterior predictive, not the plug-in posterior mean
``f_c,g = (Theta*lambda_g + C_c,g) / (Theta + N_c)``.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .model import DirichletMultinomial


@dataclass
class Summary:
    """A fitted partition as a compact generative model.  Only the K per-state
    sufficient statistics below are kept, not the original per-cell counts:
    small enough to save, share, and reload, and everything else (``freq``,
    ``hierarchy``, ``markers``, ``sample``, ...) is computed from them alone.
    """

    labels: np.ndarray
    """(N,) int, the fitted state (0..K-1) of every input cell, in input order."""
    counts: np.ndarray
    """(K, G) float, summed raw UMI counts per state: the sufficient statistic
    everything else (``freq``, ``hierarchy``, ``markers``, sampling) is
    computed from, never the original per-cell data."""
    weights: np.ndarray
    """(K,) float, sums to 1: each state's share of the input cells
    (``cluster_size / N``), the mixing proportions ``sample`` draws states
    from by default."""
    theta: float
    """The fitted Dirichlet concentration Theta (``Cluster.LAMBDA_sum``)."""
    lam: np.ndarray
    """(G,) float, sums to 1: phi, the fixed genome-wide UMI-fraction profile
    (supp. info slot 5; ``Cluster.LAMBDA / Theta``).  Shared by every state,
    not a per-state estimate."""
    lib_sizes: list
    """K 1-D int arrays: the member cells' own library sizes (total UMI) per
    state, resampled by ``sample``/``reconstruct`` for realistic per-cell
    depths instead of a single average."""
    genes: np.ndarray | None = None
    """(G,) gene names in ``counts``' column order, if given to ``run()``;
    ``None`` otherwise."""

    # ------------------------------------------------------------------ #

    @property
    def n_states(self):
        return self.counts.shape[0]

    @property
    def n_cells(self):
        return self.labels.shape[0]

    @property
    def model(self) -> DirichletMultinomial:
        """The ``DirichletMultinomial`` (Theta, phi=lam) behind this summary."""
        return DirichletMultinomial(self.theta, self.lam)

    def freq(self, kind="mean") -> np.ndarray:
        """(K, G) posterior frequency vector per state."""
        return self.model.posterior_freq(self.counts, kind=kind)

    @property
    def log_likelihood(self):
        """Total DM marginal log-likelihood of the partition: the sum of the
        per-state closed forms (supp. info eq. 15), dropping the
        partition-independent per-cell multinomial coefficients.  Comparable
        across runs on the same data at the same Theta: a partition that
        separates the cells better scores higher."""
        m = self.model
        return float(sum(m.cluster_loglik(c) for c in self.counts))

    # ------------------------------------------------------------------ #
    # generative use
    # ------------------------------------------------------------------ #

    def sample(self, n_cells, rng=None, states=None, lib_sizes=None, estimator="posterior"):
        """Draw ``n_cells`` new cells.  Returns (G, n_cells) int counts.

        states     : draw state labels from ``weights`` (default) or use these.
        lib_sizes  : per-cell library sizes; default resamples each cell's own
                     state's member library sizes.
        estimator  : how a cell's alpha is chosen, given its state.
                     "posterior" (default): a fresh alpha ~ Dirichlet(posterior
                     params) per cell before the multinomial draw, the actual
                     posterior predictive (supp. info eq. 18), properly
                     overdispersed.
                     "mean" / "mode": plug in that fixed point estimate (eq. 20
                     / eq. 19, ``Summary.freq``) for every cell of a state
                     instead. Cheaper, but every cell of a state is then
                     drawn around the identical frequency vector, understating
                     the state's own remaining alpha-uncertainty (eq. 21).
                     "mode" is sparser still (many low-count genes clip to
                     exactly 0), so its draws are typically even less spread
                     than "mean"'s.  Both plug-ins make generated cells
                     visibly tighter than real ones, e.g. in a UMAP.
        """
        rng = np.random.default_rng(rng)
        if states is None:
            states = rng.choice(self.n_states, size=n_cells, p=self.weights)
        states = np.asarray(states)
        if lib_sizes is None:
            lib_sizes = np.array([rng.choice(self.lib_sizes[c]) for c in states])
        out = np.zeros((self.counts.shape[1], n_cells), dtype=np.int64)
        if estimator == "posterior":
            a = self.model.posterior_params(self.counts)  # (K, G)
            for j in range(n_cells):
                out[:, j] = rng.multinomial(int(lib_sizes[j]), rng.dirichlet(a[states[j]]))
        elif estimator in ("mean", "mode"):
            f = self.freq(estimator)
            for j in range(n_cells):
                out[:, j] = rng.multinomial(int(lib_sizes[j]), f[states[j]])
        else:
            raise ValueError(f"estimator must be 'posterior', 'mean', or 'mode', got {estimator!r}")
        return out

    def reconstruct(self, counts, rng=None, estimator="posterior"):
        """Resample every input cell at its own library size from its state."""
        rng = np.random.default_rng(rng)
        counts = counts.tocsc() if sp.issparse(counts) else np.asarray(counts)
        L = np.asarray(counts.sum(axis=0)).ravel()
        return self.sample(
            self.n_cells, rng=rng, states=self.labels, lib_sizes=L, estimator=estimator
        )

    def predict_state(self, counts) -> np.ndarray:
        """Assign new cells (G, M) to their best-matching state under the DM
        posterior predictive.  Returns (M,) labels."""
        return self.model.assign(self.counts, counts).astype(np.int64)

    # ------------------------------------------------------------------ #
    # hierarchy / markers  (lazy: pure functions of counts + theta + lam)
    # ------------------------------------------------------------------ #

    def _state_cluster(self, n_cache=1000):
        """A K-"cell" Cluster whose cell m carries state m's summed counts.
        The merge hierarchy and marker scores depend only on the per-state count
        vectors + Theta + lambda, so this reproduces exactly what a Cluster over
        the real cells would give; no original data needed."""
        from .core import Cluster

        d = np.rint(self.counts).astype(np.int64).T  # (G, K)
        return Cluster(
            d,
            self.model.pseudocounts,
            c=np.arange(self.n_states, dtype=np.int32),
            max_clusters=self.n_states,
            n_cache=n_cache,
        )

    def hierarchy(self):
        """DataFrame of the agglomerative DM-optimal merge order of the states."""
        from .analysis import get_hierarchy_df

        return get_hierarchy_df(*self._state_cluster().get_cluster_hierarchy())

    def cut(self, n_states):
        """(N,) labels at a coarser resolution: states merged down to
        ``n_states`` along the hierarchy."""
        from .analysis import clusters_from_hierarchy, get_hierarchy_df

        hdf = get_hierarchy_df(*self._state_cluster().get_cluster_hierarchy())
        merged_state = clusters_from_hierarchy(
            hdf, cluster_init=np.arange(self.n_states), steps=max(0, self.n_states - int(n_states))
        )
        return np.asarray(merged_state)[self.labels]

    def markers(self):
        """(n_merges, G) marker-gene score table: one row per hierarchy split."""
        from .analysis import get_hierarchy_df, marker_score_table

        clst = self._state_cluster()
        hdf = get_hierarchy_df(*clst.get_cluster_hierarchy())
        return marker_score_table(clst, hdf)

    # ------------------------------------------------------------------ #
    # construction / io
    # ------------------------------------------------------------------ #

    @classmethod
    def from_cluster(cls, clst, counts):
        """Build from a converged ``Cluster`` and the (G, N) input counts."""
        labels = np.asarray(clst.clusters)
        _, labels = np.unique(labels, return_inverse=True)  # compact 0..K-1
        K = int(labels.max()) + 1
        C = np.asarray(clst.cluster_umi_counts.T, dtype=np.float64)
        C = C[np.asarray(clst.cluster_sizes) > 0]  # non-empty rows
        sizes = np.bincount(labels, minlength=K).astype(np.float64)
        lam = np.asarray(clst.LAMBDA, dtype=np.float64) / clst.LAMBDA_sum
        cnts = counts.tocsc() if sp.issparse(counts) else np.asarray(counts)
        L = np.asarray(cnts.sum(axis=0)).ravel()
        lib = [L[labels == c] for c in range(K)]
        return cls(
            labels=labels,
            counts=C,
            weights=sizes / sizes.sum(),
            theta=float(clst.LAMBDA_sum),
            lam=lam,
            lib_sizes=lib,
            genes=getattr(clst, "genes", None),
        )

    def save(self, path):
        np.savez_compressed(
            path,
            labels=self.labels,
            counts=self.counts,
            weights=self.weights,
            theta=self.theta,
            lam=self.lam,
            lib_sizes=np.array(self.lib_sizes, dtype=object),
            genes=np.array([]) if self.genes is None else self.genes,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        g = z["genes"]
        return cls(
            labels=z["labels"],
            counts=z["counts"],
            weights=z["weights"],
            theta=float(z["theta"]),
            lam=z["lam"],
            lib_sizes=list(z["lib_sizes"]),
            genes=None if g.size == 0 else g,
        )
