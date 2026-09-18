"""
The Dirichlet-multinomial model of supp. info §A1: fastcellstates' default and,
today, only model.  Single owner of the generative assumptions (the slot
numbers are ``model.base.Model``'s; see it for the general table):

    slot 2   measurement    condition on the cell total N  ->  n ~ Multinomial(N, alpha)
    slot 3   prior on alpha Dirichlet(theta)         (conjugate, rescaling-invariant)
    slot 4   concentration  theta_g = Theta * phi_g  ;  only the scalar Theta is free
    slot 5   phi            fixed genome-average profile (``model.phi.global_phi``)
    slot 6   Theta fit      Minka fixed-point MLE at a fixed partition
    slot 8   subset marginal LL   closed form, eq. 15
    slot 10  posterior predictive  mean (eq. 20) / mode (eq. 19)

The O(G) incremental move / merge deltas that make the search fast (slot 9)
are the conjugacy pay-off; for this model they live in ``model._dm_kernels``
and are driven through ``core.Cluster``.  Correctness of that fast path is
checked against ``model._dm_reference`` (pure numpy) in the test-suite.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import digamma, gammaln

from .._types import Counts
from .base import Model
from .phi import global_phi


@dataclass(frozen=True)
class DirichletMultinomial(Model):
    """A Dirichlet-multinomial expression-state model.

    Parameters
    ----------
    theta : float
        The Dirichlet concentration ``Theta`` (``Cluster.theta``).
    phi : (G,) ndarray
        The fixed profile, sums to 1 (``Cluster.phi``).
    """

    theta: float
    phi: np.ndarray

    # ---- construction ------------------------------------------------
    @classmethod
    def from_counts(cls, counts, theta, phi="global"):
        """Build with ``phi`` estimated from ``(G, N)`` counts (slot 5)."""
        p = global_phi(counts) if isinstance(phi, str) else np.asarray(phi, dtype=np.float64)
        return cls(theta=float(theta), phi=np.ascontiguousarray(p, dtype=np.float64))

    @property
    def pseudocounts(self) -> np.ndarray:
        """``theta_g = Theta * phi_g``: the Dirichlet parameter vector
        (``Cluster.dirichlet_pseudocounts``)."""
        return self.theta * self.phi

    def with_theta(self, theta):
        return DirichletMultinomial(float(theta), self.phi)

    # ---- slot 8 : a subset's marginal log-likelihood --------------
    def cluster_loglik(self, counts):
        """DM marginal LL of one subset with summed counts ``C`` (G,), eq. 15.

        Reference numpy path; ``_dm_kernels`` is the fast one and is checked
        against ``_dm_reference`` in ``test/test_kernels.py``.
        """
        C = np.asarray(counts, dtype=np.float64)
        a = self.pseudocounts
        Nc = C.sum()
        B = gammaln(self.theta) - gammaln(a).sum()
        return B - gammaln(Nc + self.theta) + gammaln(a + C).sum()

    # ---- slot 6 : Theta MLE at a fixed partition -----------------
    def fit_theta(self, state_counts, iters=500, tol=1e-6):
        """Minka fixed-point concentration MLE.

        ``state_counts`` (K, G): summed counts per non-empty subset.  Monotone
        in the likelihood; converges from either side (the depth heuristic is
        often far off on deep data).  Returns a new ``DirichletMultinomial``.
        """
        C = np.asarray(state_counts, dtype=np.float64)
        Nc = C.sum(1)
        theta = float(self.theta)
        for _ in range(iters):
            a = theta * self.phi
            num = (self.phi[None, :] * (digamma(a[None, :] + C) - digamma(a)[None, :])).sum()
            den = (digamma(theta + Nc) - digamma(theta)).sum()
            new = theta * num / den
            if not np.isfinite(new) or new <= 0:
                break
            theta_prev, theta = theta, new
            if abs(np.log(theta) - np.log(theta_prev)) < tol:
                break
        return self.with_theta(theta)

    # ---- slot 10 : posterior over a subset's alpha --------------
    def posterior_params(self, state_counts) -> np.ndarray:
        """(K, G) Dirichlet posterior parameters ``Theta*phi + C`` per subset
        (eq. 18): the full posterior, not just its mean/mode.  A fresh
        ``rng.dirichlet(a)`` draw from a row is a proper posterior sample of
        that subset's alpha; see ``Summary.sample``."""
        C = np.asarray(state_counts, dtype=np.float64)
        return self.pseudocounts[None, :] + C

    def posterior_freq(self, state_counts, kind="mean") -> np.ndarray:
        """(K, G) posterior transcription-quotient vector per subset.

        ``"mean"`` -> ``(Theta phi + C) / (Theta + N_c)``            (eq. 20)
        ``"mode"`` -> ``clip(Theta phi + C - 1, 0)`` renormalised    (eq. 19)
        """
        a = self.posterior_params(state_counts)
        if kind == "mode":
            a = np.clip(a - 1.0, 0.0, None)
        elif kind != "mean":
            raise ValueError(f"kind must be 'mean' or 'mode', got {kind!r}")
        return a / a.sum(1, keepdims=True)

    def log_posterior_predictive(self, state_counts, query: Counts) -> np.ndarray:
        """(K, M) log P(query cell | subset k) under the DM posterior predictive.

        ``query`` : ``(G, M)`` UMI counts, dense or sparse.
        """
        C = np.asarray(state_counts, dtype=np.float64)
        a = self.pseudocounts[None, :] + C  # (K, G)
        A = a.sum(1)
        lga, lgA = gammaln(a), gammaln(A)
        q = sp.csc_matrix(query)
        ind, ptr, dat = q.indices, q.indptr, q.data.astype(np.float64)
        M = int(np.asarray(q.shape)[1])
        out = np.empty((a.shape[0], M), dtype=np.float64)
        for j in range(M):
            sl = slice(ptr[j], ptr[j + 1])
            gi, gv = ind[sl], dat[sl]
            L = gv.sum()
            out[:, j] = lgA - gammaln(A + L) + (gammaln(a[:, gi] + gv[None, :]) - lga[:, gi]).sum(1)
        return out

    def assign(self, state_counts, query: Counts) -> np.ndarray:
        """(M,) argmax-posterior subset label for each query cell (G, M)."""
        return np.argmax(self.log_posterior_predictive(state_counts, query), axis=0)
