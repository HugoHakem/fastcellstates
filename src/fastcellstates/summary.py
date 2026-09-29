"""
``Summary``: the compact generative model that is the point of the pipeline.

A partition of the input cells into K states, each state a Dirichlet-multinomial
posterior over gene frequencies.  Sample new cells, reconstruct the input, or
place held-out cells, all without re-running anything.

    alpha_c ~ Dirichlet(Theta*phi + C_c)                      (posterior, eq. 18)
    cell in state c, library size L   ~   Multinomial(L, alpha_c)

a fresh alpha per sampled cell (``Summary.sample``'s default): the proper
posterior predictive, not the plug-in posterior mean
``f_c,g = (Theta*phi_g + C_c,g) / (Theta + N_c)``.

Beyond the paper (``estimate_gene_theta``, the ``gene_theta`` field and
``sample(estimator="spread")``): cellstates
treats every cell of a state as a multinomial draw of the state's one profile, so
all variation between them is counting noise.  Real cells vary more, and by an
amount that differs by gene.  ``estimate_gene_theta`` measures it as a per-gene
concentration Theta_g of a within-state spread (unrelated to the prior's Theta),
and ``sample(estimator="spread")`` draws cells with it:

    G_g ~ Gamma(Theta_g * f_c,g, scale 1/Theta_g),   alpha = G / sum(G)
    cell in state c, library size L   ~   Multinomial(L, alpha)

so alpha_g has mean f_c,g and, to first order in f, variance f_c,g / Theta_g;
Theta_g = inf is the paper's model.  The partition is fitted exactly as before.
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
    """The fitted Dirichlet concentration Theta (``Cluster.theta``)."""
    phi: np.ndarray
    """(G,) float, sums to 1: the fixed genome-wide UMI-fraction profile
    (supp. info slot 5; ``Cluster.phi``).  Shared by every state, not a
    per-state estimate."""
    lib_sizes: list
    """K 1-D int arrays: the member cells' own library sizes (total UMI) per
    state, resampled by ``sample``/``reconstruct`` for realistic per-cell
    depths instead of a single average."""
    genes: np.ndarray | None = None
    """(G,) gene names in ``counts``' column order, if given to ``run()``;
    ``None`` otherwise."""
    gene_theta: np.ndarray | None = None
    """(G,) per-gene within-state concentration (beyond the paper; see the
    module docstring), set from ``estimate_gene_theta`` and used by
    ``sample(estimator="spread")``; ``None`` until estimated."""

    # ------------------------------------------------------------------ #

    @property
    def n_states(self):
        return self.counts.shape[0]

    @property
    def n_cells(self):
        return self.labels.shape[0]

    @property
    def model(self) -> DirichletMultinomial:
        """The ``DirichletMultinomial`` (Theta, phi) behind this summary."""
        return DirichletMultinomial(self.theta, self.phi)

    def freq(self, kind="mean") -> np.ndarray:
        """(K, G) posterior frequency vector per state."""
        return self.model.posterior_freq(self.counts, kind=kind)

    def estimate_gene_theta(self, counts) -> np.ndarray:
        """(G,) per-gene within-state concentration Theta_g, by moments on the
        fitted cells (beyond the paper; see the module docstring).

        ``counts``: the (G, N) cells this summary was fitted on, in the same
        order as ``labels``.  With y = x / L a cell's share of gene g, the
        state posterior means f_c,g (``freq("mean")``) and weights w_c, the
        law of total variance over state, depth and the within-state spread
        gives, to first order in f,

            Var(y_g) = sum_c w_c (f_c,g - mu_g)^2                  between states
                     + E[1/L] sum_c w_c f_c,g (1 - f_c,g)          counting
                     + (1 - E[1/L]) mu_g / Theta_g                 within-state spread

        with mu_g = sum_c w_c f_c,g and L independent of the state.  Setting it
        to the cells' measured Var(y_g) gives Theta_g in closed form; genes no
        more variable than counting plus states get Theta_g = inf.  E[1/L] is
        the cells' own mean inverse library size.  Store it on the summary
        (``summ.gene_theta = summ.estimate_gene_theta(counts)``) to keep it
        with ``save`` and have ``sample(estimator="spread")`` use it.
        """
        cnts = counts.tocsc() if sp.issparse(counts) else np.asarray(counts)
        if cnts.shape[1] != self.n_cells:
            raise ValueError(f"counts has {cnts.shape[1]} cells, the summary {self.n_cells}")
        L = np.asarray(cnts.sum(axis=0), dtype=np.float64).ravel()
        if sp.issparse(cnts):
            y = cnts @ sp.diags(1.0 / np.maximum(L, 1.0))
            m = np.asarray(y.mean(axis=1)).ravel()
            m2 = np.asarray(y.multiply(y).mean(axis=1)).ravel()
        else:
            y = cnts / np.maximum(L, 1.0)[None, :]
            m, m2 = y.mean(axis=1), (y**2).mean(axis=1)
        n = cnts.shape[1]
        var = (m2 - m**2) * n / (n - 1)
        return self.gene_theta_from_moments(var, float(np.mean(1.0 / np.maximum(L, 1.0))))

    def gene_theta_from_moments(self, share_var, mean_inv_depth) -> np.ndarray:
        """``estimate_gene_theta`` from precomputed moments of the fitted cells,
        for populations too large to hold at once: ``share_var`` the (G,)
        variance across cells of y_g = x_g / L, ``mean_inv_depth`` the cells'
        mean of 1/L.  Both accumulate block by block (sums of y, y^2 and 1/L)."""
        var = np.asarray(share_var, dtype=np.float64)
        f = self.freq("mean")  # (K, G)
        w = self.weights
        mu = w @ f
        between = w @ (f - mu) ** 2
        counting = w @ (f * (1.0 - f))
        inv_l = float(mean_inv_depth)
        excess = var - between - inv_l * counting
        out = np.full(f.shape[1], np.inf)
        pos = excess > 0
        out[pos] = (1.0 - inv_l) * mu[pos] / excess[pos]
        return out

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

    def sample(
        self,
        n_cells,
        rng=None,
        states=None,
        lib_sizes=None,
        estimator="posterior",
        gene_theta=None,
        log_shift=None,
    ):
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
                     "spread" (beyond the paper): each cell drawn around the
                     state's posterior mean with a per-gene within-state
                     spread, ``gene_theta`` (default: the summary's own
                     ``gene_theta`` field), see the module docstring.
                     Drawn a state at a time.
        gene_theta : (G,) per-gene concentrations for "spread" (inf = no spread);
                     default the ``gene_theta`` field.
        log_shift  : (G,) optional log-fold change applied to every state's
                     frequencies before the draw, alpha -> normalize(alpha *
                     exp(log_shift)): the same population, perturbed.
        """
        rng = np.random.default_rng(rng)
        if states is None:
            states = rng.choice(self.n_states, size=n_cells, p=self.weights)
        states = np.asarray(states)
        if lib_sizes is None:
            lib_sizes = np.array([rng.choice(self.lib_sizes[c]) for c in states])
        lib_sizes = np.asarray(lib_sizes)
        out = np.zeros((self.counts.shape[1], n_cells), dtype=np.int64)
        shift = None if log_shift is None else np.exp(np.asarray(log_shift, dtype=np.float64))

        def shifted(a):
            if shift is None:
                return a
            a = a * shift
            return a / a.sum(axis=-1, keepdims=True)

        if estimator == "spread":
            if gene_theta is None:
                gene_theta = self.gene_theta
            if gene_theta is None:
                raise ValueError(
                    'estimator="spread" needs gene_theta (pass it, or set it from estimate_gene_theta)'
                )
            tg = np.asarray(gene_theta, dtype=np.float64)
            fin = np.isfinite(tg)
            f = shifted(self.freq("mean"))
            for c in np.unique(states):
                cells = np.flatnonzero(states == c)
                g = np.repeat(f[c][None, :], len(cells), axis=0)
                g[:, fin] = (
                    rng.standard_gamma(tg[fin] * f[c, fin], size=(len(cells), int(fin.sum())))
                    / tg[fin]
                )
                g /= g.sum(axis=1, keepdims=True)
                out[:, cells] = rng.multinomial(np.asarray(lib_sizes[cells], dtype=np.int64), g).T
            return out
        if estimator == "posterior":
            a = self.model.posterior_params(self.counts)  # (K, G)
            for j in range(n_cells):
                out[:, j] = rng.multinomial(int(lib_sizes[j]), shifted(rng.dirichlet(a[states[j]])))
        elif estimator in ("mean", "mode"):
            f = shifted(self.freq(estimator))
            for j in range(n_cells):
                out[:, j] = rng.multinomial(int(lib_sizes[j]), f[states[j]])
        else:
            raise ValueError(
                f"estimator must be 'posterior', 'mean', 'mode' or 'spread', got {estimator!r}"
            )
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
    # hierarchy / markers  (lazy: pure functions of counts + theta + phi)
    # ------------------------------------------------------------------ #

    def _state_cluster(self, n_cache=1000):
        """A K-"cell" Cluster whose cell m carries state m's summed counts.
        The merge hierarchy and marker scores depend only on the per-state count
        vectors and the Dirichlet prior (theta, phi), so this reproduces exactly
        what a Cluster over the real cells would give; no original data needed."""
        from .core import Cluster

        d = np.rint(self.counts).astype(np.int64).T  # (G, K)
        return Cluster(
            d,
            pseudocounts=self.model.pseudocounts,
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
        cnts = counts.tocsc() if sp.issparse(counts) else np.asarray(counts)
        L = np.asarray(cnts.sum(axis=0)).ravel()
        lib = [L[labels == c] for c in range(K)]
        return cls(
            labels=labels,
            counts=C,
            weights=sizes / sizes.sum(),
            theta=float(clst.theta),
            phi=clst.phi,
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
            phi=self.phi,
            lib_sizes=np.array(self.lib_sizes, dtype=object),
            genes=np.array([]) if self.genes is None else self.genes,
            gene_theta=np.array([]) if self.gene_theta is None else self.gene_theta,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        g = z["genes"]
        tg = z["gene_theta"] if "gene_theta" in z.files else np.array([])  # absent from older files
        return cls(
            labels=z["labels"],
            counts=z["counts"],
            weights=z["weights"],
            theta=float(z["theta"]),
            phi=z["phi"],
            lib_sizes=list(z["lib_sizes"]),
            genes=None if g.size == 0 else g,
            gene_theta=None if tg.size == 0 else tg,
        )
