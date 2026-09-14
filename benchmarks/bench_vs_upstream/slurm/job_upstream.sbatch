#!/usr/bin/env bash
#SBATCH --job-name=fcs-bench-upstream
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=benchmarks/bench_vs_upstream/results/slurm-%x-%j.out
#
# One original-cellstates run at one thread count. --cpus-per-task is fixed
# at 16 regardless of --threads (matching every other job in this
# comparison), not matched to it: cellstates's own OpenMP num_threads caps
# how many threads it actually uses, so idle allocated cores beyond that
# don't change its measured behaviour, and a uniform allocation across every
# job is simpler than sizing each one individually. $1 is --threads,
# remaining args forward to run_upstream.py (--data/--out).
#   sbatch --job-name=fcs-bench-upstream-t4 job_upstream.sbatch 4 \
#       --out benchmarks/bench_vs_upstream/results/upstream_t4.json
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
THREADS="$1"; shift
pixi run --manifest-path benchmarks/bench_vs_upstream/upstream_cellstates/pixi.toml python \
    benchmarks/bench_vs_upstream/scripts/run_upstream.py --threads "$THREADS" "$@"
