#!/usr/bin/env bash
#SBATCH --job-name=fcs-bench-prepare
#SBATCH --partition=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=benchmarks/bench_vs_upstream/results/slurm-prepare-%j.out
#
# Submitted by submit_jobs.sh; extra args forward to prepare_data.py
# (--dataset / --h5ad / --out).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p benchmarks/bench_vs_upstream/results
pixi run python benchmarks/bench_vs_upstream/scripts/prepare_data.py "$@"
