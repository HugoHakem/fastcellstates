# Notebooks

Worked examples for `fastcellstates`.

- **`getting_started.ipynb`** — loads `scanpy.datasets.pbmc3k` (raw UMI
  counts), runs the `fast` preset, and walks through the resulting `Summary`:
  labels, frequencies, mixing weights, the merge hierarchy, marker genes,
  `.sample()` / `.reconstruct()` / `.predict_state()`, a UMAP of real vs.
  freshly-sampled cells in the standard log-normalised PCA space, tuning
  `fast` against the `exact` reference, save/load, and a dendrogram plot.
  Needs the `notebook` dependency group (`scanpy`, `ipykernel`, `jupyterlab`)
  and the `plot` extra for the last cell.
- **`large_datasets.ipynb`** — the pattern for a dataset too large to fit in
  memory: fit a `Summary` on a subsample, then stream-assign the rest with
  `Summary.predict_state`, reading chunks straight from disk via
  `io.peek_h5ad` / `io.read_h5ad_subset` / `io.iter_h5ad_chunks` (AnnData
  `backed="r"` mode under the hood). Uses `pbmc3k` too, purely for
  familiarity and speed; it doesn't actually need this pattern itself.

Run either: `pixi run jupyter lab docs/notebooks/<name>.ipynb`.
