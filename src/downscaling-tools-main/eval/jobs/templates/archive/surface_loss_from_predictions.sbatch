#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
# Weighted surface loss from completed prediction NetCDFs.

#SBATCH --job-name=surface_loss_from_predictions
#SBATCH --qos=ng
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=0
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

RUN_ROOT="/home/ecm5702/scratch/eval/manual_example"
RUN_ID="manual_example"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"
# Canonical output path (eval-data-contract)
OUT_JSON="${RUN_ROOT}/data/surface_loss.json"
PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
SURFACE_SCRIPT="${PROJECT_ROOT}/eval/jobs/scoreboard_surface_loss.py"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*)
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag*)
    source /home/ecm5702/dev/.ds-ag/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
  *)
    die "Unknown cluster from hostname '${HOST_SHORT}'."
    ;;
esac

[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"
[[ -f "${SURFACE_SCRIPT}" ]] || die "Missing surface loss script: ${SURFACE_SCRIPT}"

mkdir -p "${RUN_ROOT}/data"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "[INFO] host=${HOST_SHORT}"
echo "[INFO] run_id=${RUN_ID}"
echo "[INFO] predictions_dir=${PREDICTIONS_DIR}"
echo "[INFO] out_json=${OUT_JSON}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

"${PYTHON}" "${SURFACE_SCRIPT}" \
  --predictions-dir "${PREDICTIONS_DIR}" \
  --out-json "${OUT_JSON}"

[[ -f "${OUT_JSON}" ]] || die "Expected surface-loss summary missing: ${OUT_JSON}"

# Legacy compat symlink
ln -sfn "data/surface_loss.json" "${RUN_ROOT}/surface_loss_summary.json"

echo "DONE: ${OUT_JSON}"
