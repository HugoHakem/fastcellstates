#!/usr/bin/env bash
#SBATCH --job-name=fcs-bench-render
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=benchmarks/bench_vs_upstream/results/slurm-render-%j.out
#
# Submitted last by submit_jobs.sh, depending on every other job.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
pixi run python benchmarks/bench_vs_upstream/scripts/render_table.py \
    --out docs/_static/benchmark_vs_upstream.svg
