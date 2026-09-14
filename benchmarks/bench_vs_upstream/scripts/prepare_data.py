"""
Fetch a dataset and save its (G, N) integer counts as a dense .npy: the one
format both fastcellstates and the original cellstates read identically, so
both sides of the comparison cluster the exact same matrix.

    pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py
    pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py \
        --dataset pbmc3k --out benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy
    # or point it at your own .h5ad:
    pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py \
        --h5ad /path/to/data.h5ad --out benchmarks/bench_vs_upstream/_data/mine.npy
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DEFAULT_OUT = "benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy"


def _load_pbmc3k():
    import scanpy as sc

    sc.settings.verbosity = 0
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    return adata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pbmc3k"], default="pbmc3k", help="a built-in dataset")
    ap.add_argument("--h5ad", default=None, help="or: path to your own .h5ad (cells x genes)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.h5ad:
        import anndata

        adata = anndata.read_h5ad(args.h5ad)
    elif args.dataset == "pbmc3k":
        adata = _load_pbmc3k()
    else:
        raise ValueError(args.dataset)

    if adata.X is None:
        raise ValueError("adata.X is empty")
    X = adata.X.T  # genes x cells
    counts = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.int32)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, counts)
    print(f"wrote {out}: {counts.shape[0]} genes x {counts.shape[1]} cells")


if __name__ == "__main__":
    main()
