"""
Shared type aliases.

``scipy.sparse`` ships only partial inline types: the abstract ``spmatrix`` /
``sparray`` bases don't expose their methods to a type checker, but the
*concrete* classes do.  So the "counts matrix" alias lists the concrete sparse
formats we actually see (``.mtx`` -> coo, ``.h5ad`` -> csr, internal -> csc),
and code coerces with ``sp.csc_matrix(x)`` (the constructor accepts dense or
sparse and is typed) rather than ``x.tocsc()`` on a maybe-``ndarray``.
"""

import numpy as np
import scipy.sparse as sp

Sparse = sp.csc_matrix | sp.csr_matrix | sp.coo_matrix | sp.csc_array | sp.csr_array | sp.coo_array
Counts = np.ndarray | Sparse
