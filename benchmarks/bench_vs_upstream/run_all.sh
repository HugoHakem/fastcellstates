#!/usr/bin/env bash
# Orchestrates the comparison: prepares the shared data once, runs each
# selected config in its own pixi environment (fastcellstates's own env for
# exact/fast/fast_res1, benchmarks/bench_vs_upstream/upstream_cellstates for the original),
# then renders the results table. Each step is also just a standalone
# script (see run_fastcellstates.py / run_upstream.py / prepare_data.py)
# if you only want to rerun one thing directly.
#
#   bash benchmarks/bench_vs_upstream/run_all.sh
#   bash benchmarks/bench_vs_upstream/run_all.sh --only fast,upstream
#   bash benchmarks/bench_vs_upstream/run_all.sh --upstream-threads 1,4,16
#   bash benchmarks/bench_vs_upstream/run_all.sh --dataset pbmc3k \
#       --data benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy
#   bash benchmarks/bench_vs_upstream/run_all.sh --h5ad /path/to/data.h5ad \
#       --data benchmarks/bench_vs_upstream/_data/mine.npy
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ONLY="exact,fast,fast_res1,upstream"
UPSTREAM_THREADS="1"
DATASET="pbmc3k"
H5AD=""
DATA="benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --upstream-threads) UPSTREAM_THREADS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --h5ad) H5AD="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

RESULTS=benchmarks/bench_vs_upstream/results
mkdir -p "$RESULTS"
IFS=',' read -ra STEPS <<< "$ONLY"
_has() { printf '%s\n' "${STEPS[@]}" | grep -qx "$1"; }

if _has exact || _has fast || _has fast_res1 || _has upstream; then
    echo "== preparing data =="
    if [[ -n "$H5AD" ]]; then
        pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py --h5ad "$H5AD" --out "$DATA"
    else
        pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py --dataset "$DATASET" --out "$DATA"
    fi
fi

if _has exact; then
    echo "== fastcellstates: exact =="
    pixi run python benchmarks/bench_vs_upstream/scripts/run_fastcellstates.py \
        --data "$DATA" --config exact --out "$RESULTS/exact.json"
fi

if _has fast; then
    echo "== fastcellstates: fast =="
    pixi run python benchmarks/bench_vs_upstream/scripts/run_fastcellstates.py \
        --data "$DATA" --config fast --out "$RESULTS/fast.json"
fi

if _has fast_res1; then
    echo "== fastcellstates: fast @ resolution=1.0 =="
    pixi run python benchmarks/bench_vs_upstream/scripts/run_fastcellstates.py \
        --data "$DATA" --config fast_res1 --out "$RESULTS/fast_res1.json"
fi

if _has upstream; then
    IFS=',' read -ra TLIST <<< "$UPSTREAM_THREADS"
    for t in "${TLIST[@]}"; do
        echo "== original cellstates: threads=$t =="
        pixi run --manifest-path benchmarks/bench_vs_upstream/upstream_cellstates/pixi.toml python \
            benchmarks/bench_vs_upstream/scripts/run_upstream.py \
            --data "$DATA" --threads "$t" --out "$RESULTS/upstream_t${t}.json"
    done
fi

echo "== rendering table =="
pixi run python benchmarks/bench_vs_upstream/scripts/render_table.py \
    --out docs/_static/benchmark_vs_upstream.svg
