#!/usr/bin/env bash
# Submits the comparison as independent SLURM jobs (not one serial script):
# prepare -> {fastcellstates configs, upstream thread counts} in parallel ->
# render, chained with --dependency so nothing starts before its data
# exists and the table only renders once everything else has finished.
# Every job requests the same --cpus-per-task (set in the slurm/job_*.sbatch
# files, currently 16): each tool's own thread setting still caps how many
# it actually uses, so unused allocated cores don't change what's measured,
# and one consistent allocation across every job is simpler than sizing
# each individually.
#
# Every job is constrained (--constraint) to one *homogeneous hardware
# family* (same vendor/CPU/RAM feature tags), auto-picked as the largest
# such family on the requested partition unless --node-constraint is given.
# That controls the real confound (different node families here span Intel
# Ice Lake vs. AMD, and different core/RAM sizes) while letting jobs land
# on different physical machines within that family, so they don't compete
# with each other for one node's shared memory bandwidth/cache. Pinning to
# one literal node instead would remove hardware variation entirely but add
# same-node contention between our own jobs when they run in parallel;
# for the size of differences this comparison actually measures, a
# same-model-CPU family is the better trade-off.
#
#   bash benchmarks/bench_vs_upstream/submit_jobs.sh
#   bash benchmarks/bench_vs_upstream/submit_jobs.sh --only fast,upstream
#   bash benchmarks/bench_vs_upstream/submit_jobs.sh --upstream-threads 1,4,8,16
#   bash benchmarks/bench_vs_upstream/submit_jobs.sh --node-constraint "lenovo&amd&local_900g"
#   bash benchmarks/bench_vs_upstream/submit_jobs.sh --h5ad /path/to/data.h5ad \
#       --data benchmarks/bench_vs_upstream/_data/mine.npy
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

ONLY="exact,fast,fast_res1,upstream"
UPSTREAM_THREADS="1,4,8,16"
DATASET="pbmc3k"
H5AD=""
DATA="benchmarks/bench_vs_upstream/_data/pbmc3k_counts.npy"
PARTITION="standard"
CONSTRAINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --upstream-threads) UPSTREAM_THREADS="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --h5ad) H5AD="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --node-constraint) CONSTRAINT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CONSTRAINT" ]]; then
    # the largest schedulable (idle/mixed) same-feature node family
    FEATURES=$(sinfo -h -p "$PARTITION" -t idle,mixed -o "%D %f" | sort -rn | head -1 | cut -d' ' -f2)
    if [[ -z "$FEATURES" ]]; then
        echo "no schedulable node family found on partition $PARTITION; pass --node-constraint explicitly" >&2
        exit 1
    fi
    CONSTRAINT="${FEATURES//,/&}"
fi
echo "constraining every job to the node family: $CONSTRAINT"
SB=(sbatch --parsable --partition="$PARTITION" --constraint="$CONSTRAINT")

RESULTS=benchmarks/bench_vs_upstream/results
mkdir -p "$RESULTS"
IFS=',' read -ra STEPS <<< "$ONLY"
_has() { printf '%s\n' "${STEPS[@]}" | grep -qx "$1"; }

if [[ -n "$H5AD" ]]; then
    PREP_ARGS=(--h5ad "$H5AD" --out "$DATA")
else
    PREP_ARGS=(--dataset "$DATASET" --out "$DATA")
fi
PREP_ID=$("${SB[@]}" benchmarks/bench_vs_upstream/slurm/job_prepare.sbatch "${PREP_ARGS[@]}")
echo "prepare: $PREP_ID"

ALL_IDS=("$PREP_ID")

if _has exact; then
    ID=$("${SB[@]}" --job-name=fcs-bench-exact --dependency=afterok:"$PREP_ID" \
        benchmarks/bench_vs_upstream/slurm/job_fastcellstates.sbatch exact \
        --data "$DATA" --out "$RESULTS/exact.json")
    echo "fastcellstates exact: $ID"
    ALL_IDS+=("$ID")
fi

if _has fast; then
    ID=$("${SB[@]}" --job-name=fcs-bench-fast --dependency=afterok:"$PREP_ID" \
        benchmarks/bench_vs_upstream/slurm/job_fastcellstates.sbatch fast \
        --data "$DATA" --out "$RESULTS/fast.json")
    echo "fastcellstates fast: $ID"
    ALL_IDS+=("$ID")
fi

if _has fast_res1; then
    ID=$("${SB[@]}" --job-name=fcs-bench-fast_res1 --dependency=afterok:"$PREP_ID" \
        benchmarks/bench_vs_upstream/slurm/job_fastcellstates.sbatch fast_res1 \
        --data "$DATA" --out "$RESULTS/fast_res1.json")
    echo "fastcellstates fast_res1: $ID"
    ALL_IDS+=("$ID")
fi

if _has upstream; then
    IFS=',' read -ra TLIST <<< "$UPSTREAM_THREADS"
    for t in "${TLIST[@]}"; do
        ID=$("${SB[@]}" --job-name="fcs-bench-upstream-t${t}" \
            --dependency=afterok:"$PREP_ID" \
            benchmarks/bench_vs_upstream/slurm/job_upstream.sbatch "$t" \
            --data "$DATA" --out "$RESULTS/upstream_t${t}.json")
        echo "upstream threads=$t: $ID"
        ALL_IDS+=("$ID")
    done
fi

DEP=$(IFS=:; echo "${ALL_IDS[*]}")
RENDER_ID=$("${SB[@]}" --dependency=afterok:"$DEP" benchmarks/bench_vs_upstream/slurm/job_render.sbatch)
echo "render: $RENDER_ID"

echo
echo "all jobs constrained to: $CONSTRAINT; track with: squeue -u \$USER -o '%.10i %.25j %.8T %.10M %N %R'"
