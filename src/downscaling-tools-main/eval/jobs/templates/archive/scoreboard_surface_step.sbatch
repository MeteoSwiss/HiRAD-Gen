#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
#SBATCH --job-name=scoreboard_surface
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail
RUN_ID="REPLACE_RUN_ID"
OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_ROOT=""
PRED_DIR=""
EXPECTED_PRED_FILES=25
[[ "${RUN_ID}" != REPLACE_* ]] || { echo "ERROR: set RUN_ID" >&2; exit 2; }
[[ -n "${RUN_ROOT}" ]] || RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ -n "${PRED_DIR}" ]] || PRED_DIR="${RUN_ROOT}/predictions"
REPO_ROOT="${DOWNSCALING_TOOLS_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
SFC_OUT="${RUN_ROOT}/data/surface_loss.json"
mkdir -p "${RUN_ROOT}/data"
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python "${REPO_ROOT}/eval/jobs/scoreboard_surface_loss.py" --predictions-dir "${PRED_DIR}" --out-json "${SFC_OUT}"
ln -sfn "data/surface_loss.json" "${RUN_ROOT}/surface_loss_summary.json" 2>/dev/null || true
echo "DONE scoreboard_surface_step run_id=${RUN_ID}"
echo "SFC_SUMMARY=${SFC_OUT}"
