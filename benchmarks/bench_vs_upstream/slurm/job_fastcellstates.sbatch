#!/usr/bin/env bash
#SBATCH --job-name=fcs-bench-fcs
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=benchmarks/bench_vs_upstream/results/slurm-%x-%j.out
#
# One fastcellstates config. Submitted by submit_jobs.sh with --job-name
# set per config for readable squeue output; $1 is the --config value,
# remaining args forward to run_fastcellstates.py (--data/--out).
#   sbatch --job-name=fcs-bench-fast job_fastcellstates.sbatch fast \
#       --out benchmarks/bench_vs_upstream/results/fast.json
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CONFIG="$1"; shift
pixi run python benchmarks/bench_vs_upstream/scripts/run_fastcellstates.py --config "$CONFIG" "$@"
