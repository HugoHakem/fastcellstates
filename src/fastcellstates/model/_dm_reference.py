"""
The Dirichlet-multinomial partition likelihood, written slowly and obviously
in pure numpy: the executable *spec* for ``DirichletMultinomial`` and the
fast ``_dm_kernels``.  ``test/test_kernels.py`` checks the fast path against
this; ``test/oracle.py`` re-exports it for back-compat.

    LL_c = B - lgamma(N_c + Theta) + sum_g lgamma(pseudocounts_g + C_c,g)
    B    = lgamma(Theta) - sum_g lgamma(pseudocounts_g)   (empty cluster -> LL_c = 0)

with Theta = sum_g pseudocounts_g (``= Theta * phi_g`` per gene) and C_c,g the
summed UMI counts of cluster c (supp. info eq. 15).
"""

import numpy as np
from scipy.special import gammaln


def partition_loglik(counts, labels, pseudocounts):
    """counts: (G, N) int ; labels: (N,) ; pseudocounts: (G,).  Returns (total, per_cluster_dict)."""
    counts = np.asarray(counts)
    labels = np.asarray(labels)
    pseudocounts = np.asarray(pseudocounts, dtype=np.float64)
    theta = pseudocounts.sum()
    B = gammaln(theta) - gammaln(pseudocounts).sum()
    per = {}
    total = 0.0
    for c in np.unique(labels):
        cols = labels == c
        if not cols.any():
            per[int(c)] = 0.0
            continue
        C = counts[:, cols].sum(axis=1).astype(np.float64)
        Nc = C.sum()
        LL = B - gammaln(Nc + theta) + gammaln(pseudocounts + C).sum()
        per[int(c)] = LL
        total += LL
    return total, per


def merge_delta(counts, labels, i, j, pseudocounts):
    """Change in total log-likelihood from merging clusters i and j."""
    _, per0 = partition_loglik(counts, labels, pseudocounts)
    merged = labels.copy()
    merged[merged == j] = i
    _, per1 = partition_loglik(counts, merged, pseudocounts)
    return per1[i] - per0.get(i, 0.0) - per0.get(j, 0.0)
