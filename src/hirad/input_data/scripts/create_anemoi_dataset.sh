#!/bin/bash
# Build an anemoi zarr dataset from a recipe: init once, then load it in
# parallel parts, then finalise + cleanup. See the anemoi-dataset-create
# skill for background.
#
# MARS retrieval is network I/O against ECMWF, not GPU compute, so this runs
# directly on the login node -- not through SLURM.
#
# `anemoi-datasets create --processes/--threads` is NOT used here: as of
# anemoi-datasets 0.5.41, its internal parallel_create() crashes for the
# `trajectories` layout (task_init returns None for the part count, then
# `for n in range(total)` raises TypeError). The init/load/finalise path
# below sidesteps that -- we pick N ourselves instead of relying on it.
#
# Environment-agnostic: if `anemoi-datasets` is already on PATH (e.g. inside
# the SLURM container -- see create_anemoi_dataset_slurm.sh) it's used as-is;
# otherwise falls back to the local dev venv, for direct login-node use.
#
# Usage:
#   create_anemoi_dataset.sh <recipe.yaml> <output.zarr> [n_parts] [log_dir]
#
# Example (login node):
#   nohup src/hirad/input_data/scripts/create_anemoi_dataset.sh \
#       src/hirad/input_data/configs/medrange.yaml \
#       /capstor/scratch/cscs/mmcgloho/anemoitest/leadtime1h.zarr \
#       12 \
#       > create.log 2>&1 &

set -euo pipefail

RECIPE=$1
OUTPUT=$2
N_PARTS=${3:-8}
LOG_DIR=${4:-./logs/$(basename "$OUTPUT" .zarr)}

if ! command -v anemoi-datasets >/dev/null 2>&1; then
    source ~/hirad-gen/HiRAD-Gen/.venv/bin/activate
fi
mkdir -p "$LOG_DIR"

# This workload is network-bound (waiting on MARS), not compute-bound. Each
# part is a separate process that would otherwise auto-detect the login
# node's full CPU count and try to spin up that many BLAS threads; with
# N_PARTS processes doing that concurrently they can exceed the login node's
# thread/process ceiling and spam "pthread_create failed" warnings (harmless,
# but noisy). Cap it instead.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date)] init: $RECIPE -> $OUTPUT"
anemoi-datasets init --overwrite "$RECIPE" "$OUTPUT" 2>&1 | tee "$LOG_DIR/init.log"

echo "[$(date)] loading $N_PARTS parts in parallel (logs: $LOG_DIR/load_*.log)"
pids=()
for i in $(seq 1 "$N_PARTS"); do
    anemoi-datasets load "$OUTPUT" --parts "$i/$N_PARTS" > "$LOG_DIR/load_${i}.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "[$(date)] one or more parts failed -- check $LOG_DIR/load_*.log" >&2
    exit 1
fi

echo "[$(date)] finalise"
anemoi-datasets finalise "$OUTPUT" 2>&1 | tee "$LOG_DIR/finalise.log"

echo "[$(date)] cleanup"
anemoi-datasets cleanup "$OUTPUT" 2>&1 | tee "$LOG_DIR/cleanup.log"

echo "[$(date)] done -- inspect with: anemoi-datasets inspect $OUTPUT --detailed --statistics"
