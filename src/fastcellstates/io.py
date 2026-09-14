"""
Read UMI count data into a genes x cells matrix.

``.h5ad`` / ``.mtx`` stay sparse (never densified); ``.txt/.tsv/.csv/.npy`` are
dense.  ``read()`` concatenates multiple files column-wise (genes assumed to
match, in order); it fully materialises whatever it returns.

For an ``.h5ad`` too large to hold in memory at once: ``peek_h5ad`` reads
shape/genes/cells without touching ``X``, ``read_h5ad_subset`` materialises
only a given set of cells, and ``iter_h5ad_chunks`` streams the whole file a
chunk at a time.  All three go through anndata's ``backed="r"`` mode (a
lazy reference to the on-disk arrays, not a load), so none of them pays for
the cells they don't touch.  The intended pattern for data too big to
cluster whole: fit a ``Summary`` on a subsample (``read_h5ad_subset``), then
``Summary.predict_state`` the rest chunk by chunk (``iter_h5ad_chunks``);
see ``docs/notebooks/large_datasets.ipynb``.
"""

import numpy as np
import scipy.sparse as sp

from ._types import Counts

_Names = np.ndarray | None


def _ncols(x: Counts) -> int:
    """x.shape[1] (scipy sparse `.shape` is typed Optional upstream)."""
    return int(np.asarray(x.shape)[1])


def _read_one(path) -> tuple[Counts, _Names, _Names]:
    """path -> (data (G, N), genes | None, cells | None)."""
    ext = path.split(".")[-1].lower()
    if ext in ("txt", "tsv", "csv", "zip", "gz", "bz2", "xz"):
        import pandas as pd

        df = pd.read_csv(path, sep=r"\s+", header=0, index_col=0)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=None, header=0, index_col=0, engine="python")
        return df.to_numpy(), df.index.to_numpy(), df.columns.to_numpy()
    if ext == "npy":
        return np.load(path), None, None
    if ext == "mtx":
        import scipy.io as sio

        return sp.csc_matrix(sio.mmread(path)), None, None
    if ext == "h5ad":
        import anndata

        a = anndata.read_h5ad(path)
        if a.X is None:
            raise ValueError(f"{path}: adata.X is empty")
        d = sp.csc_matrix(a.X).T  # obs x var -> genes x cells
        return d, np.asarray(a.var_names), np.asarray(a.obs_names)
    raise ValueError(f"unrecognised file type: {path}")


def _to_int(data: Counts) -> Counts:
    if sp.issparse(data):
        m = sp.csc_matrix(data)
        if np.issubdtype(m.dtype, np.floating):
            m.data = np.rint(m.data)
        return m.astype(np.int64)
    a = np.asarray(data)
    if np.issubdtype(a.dtype, np.floating):
        a = np.rint(a)
    return a.astype(np.int64, copy=False)


def read(paths, gene_mask_zero=False):
    """One path or a list -> (counts (G, N) int, genes, cells).

    Multiple files are stacked column-wise.  Any file that is sparse makes the
    result sparse (CSC).  ``gene_mask_zero`` drops all-zero genes (the Cluster
    does this internally too; use it only when you need the mask up front).
    """
    if isinstance(paths, (str, bytes)):
        paths = [paths]
    datas: list[Counts] = []
    cells: list[np.ndarray] = []
    genes: _Names = None
    for p in paths:
        d, g, c = _read_one(p)
        if g is not None:
            genes = g
        datas.append(_to_int(d))
        n = _ncols(d)
        cells.append(c if c is not None else np.array([f"{p}-cell_{i}" for i in range(n)]))

    if any(sp.issparse(d) for d in datas):
        counts: Counts = sp.hstack([sp.csc_matrix(d) for d in datas], format="csc")
    else:
        counts = np.concatenate([np.asarray(d) for d in datas], axis=1)
    cell_names = np.concatenate(cells)

    if gene_mask_zero:
        keep = np.asarray(counts.sum(axis=1)).ravel() > 0
        counts = sp.csc_matrix(counts)[keep] if sp.issparse(counts) else np.asarray(counts)[keep]
        if genes is not None:
            genes = genes[keep]
    return counts, genes, cell_names


# --------------------------------------------------------------------------- #
# backed access to a large .h5ad: never materialise cells you don't touch
# --------------------------------------------------------------------------- #


def peek_h5ad(path: str) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """``((G, N), genes, cells)`` of an ``.h5ad`` file without loading ``X``:
    for sizing up a file too large to load whole.  The shape order matches
    this module's own (genes, cells) convention, the opposite of AnnData's
    own (obs, var)."""
    import anndata

    a = anndata.read_h5ad(path, backed="r")
    n_obs, n_var = a.shape
    return (n_var, n_obs), np.asarray(a.var_names), np.asarray(a.obs_names)


def read_h5ad_subset(path: str, cell_idx) -> tuple[Counts, np.ndarray, np.ndarray]:
    """(G, len(cell_idx)) counts for just the given cells of an ``.h5ad``
    file, via anndata's backed mode: the rest of the file is never
    materialised.  ``cell_idx``: int array, indices into ``adata.obs``
    (any order; used to pull a fitting subsample out of a file too large to
    cluster whole)."""
    import anndata

    a = anndata.read_h5ad(path, backed="r")
    sub = a[np.asarray(cell_idx), :]
    if sub.X is None:
        raise ValueError(f"{path}: adata.X is empty")
    d = sp.csc_matrix(sub.X[:]).T  # obs x var -> genes x cells
    return _to_int(d), np.asarray(a.var_names), np.asarray(sub.obs_names)


def iter_h5ad_chunks(path: str, chunk_size: int = 5000):
    """Yield an ``.h5ad`` file's cells as (G, chunk_size) chunks (the last
    chunk may be smaller), one chunk materialised at a time via anndata's
    backed mode: for scoring a dataset too large to hold in memory at once,
    e.g. ``Summary.predict_state`` after fitting on a ``read_h5ad_subset``
    sample.  Yields ``(counts_chunk, genes, cell_names_chunk)``."""
    import anndata

    a = anndata.read_h5ad(path, backed="r")
    genes = np.asarray(a.var_names)
    for start in range(0, a.n_obs, chunk_size):
        sub = a[start : start + chunk_size, :]
        if sub.X is None:
            raise ValueError(f"{path}: adata.X is empty")
        d = sp.csc_matrix(sub.X[:]).T
        yield _to_int(d), genes, np.asarray(sub.obs_names)
