"""
The fixed genome-average expression profile ``phi`` (supp. info §A1, slot 5;
see ``model.base.Model`` for the slot table).

    phi_g = (sum_c n_gc) / (sum_{c,g} n_gc)

i.e. the optimal Dirichlet direction for the trivial one-cluster partition
(eq. 12).  ``phi`` is fixed for a dataset; only the concentration ``Theta`` is
then optimised.
"""

import numpy as np
import scipy.sparse as sp


def global_phi(counts):
    """(G,) genome-wide UMI fractions from ``(G, N)`` counts (dense or sparse)."""
    counts = counts.tocsr() if sp.issparse(counts) else np.asarray(counts)
    g = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    tot = g.sum()
    if tot <= 0:
        raise ValueError("all counts are zero, cannot estimate phi")
    return g / tot
