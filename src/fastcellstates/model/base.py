"""
The model contract.

A ``Model`` owns the generative assumptions behind the partition likelihood
(supp. info §A1).  The search blocks (``graph`` / ``partition`` / ``moves``)
and ``summary`` consult it; they never hard-code a distribution.

Modelling slots (numbered as in the paper's derivation; slot 1, mRNA counts
as a Poisson process given the transcription rate, eq. 1-2, is background
physics, not a choice ``Model`` varies, so the table starts at 2):

===  ==================================  ==================================
 2   measurement / capture                3   prior family on alpha
 4   concentration parametrisation        5   the fixed profile phi
 6   hyper-parameter fitting              7   partition prior P(rho)
 8   a subset's marginal log-likelihood   9   incremental move / merge deltas
10   posterior predictive                11   per-gene marker decomposition
===  ==================================  ==================================

Only ``DirichletMultinomial`` implements this today.  Slot 9, the O(G)
incremental deltas that make the search fast, is the pay-off of Dirichlet
conjugacy (slot 3); for the DM model it is driven through ``core.Cluster`` +
``model._dm_kernels`` rather than the methods below.  A non-conjugate model
would have to supply its own move machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Generative model for a partition of cells into expression states."""

    @abstractmethod
    def cluster_loglik(self, counts) -> float:
        """Marginal log-likelihood of one subset from its summed counts (G,); slot 8."""

    @abstractmethod
    def fit_theta(self, state_counts) -> Model:
        """Return a copy with the hyper-parameter(s) re-fit at a fixed partition; slot 6.

        ``state_counts`` is (K, G): summed UMI counts per non-empty subset.
        """

    @abstractmethod
    def posterior_freq(self, state_counts, kind="mean") -> np.ndarray:
        """(K, G) posterior expression-state vector per subset; slot 10."""

    @abstractmethod
    def log_posterior_predictive(self, state_counts, query) -> np.ndarray:
        """(K, M) log P(query cell | subset) for query counts ``(G, M)``; slot 10."""
