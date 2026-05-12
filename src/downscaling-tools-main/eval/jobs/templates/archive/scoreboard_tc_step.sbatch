#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
#SBATCH --job-name=scoreboard_tc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail
RUN_ID="REPLACE_RUN_ID"
LANE="o96_o320"
OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_ROOT=""
PRED_DIR=""
EXPECTED_PRED_FILES=25
TC_EVENTS="idalia,franklin"

[[ "${RUN_ID}" != REPLACE_* ]] || { echo "ERROR: set RUN_ID" >&2; exit 2; }
[[ -n "${RUN_ROOT}" ]] || RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ -n "${PRED_DIR}" ]] || PRED_DIR="${RUN_ROOT}/predictions"
REPO_ROOT="${DOWNSCALING_TOOLS_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
TC_EVENT_TAG="${TC_EVENTS//,/__}"
mkdir -p "${RUN_ROOT}/data/tc" "${RUN_ROOT}/plots/tc"
TC_STATS="${RUN_ROOT}/data/tc/${TC_EVENT_TAG}.stats.json"
TC_PDF="${RUN_ROOT}/plots/tc/${TC_EVENT_TAG}.pdf"

source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
python -m eval.cli evaluate --predictions-dir "${PRED_DIR}" --lane "${LANE}" --only tc --overwrite
if [[ -f "${RUN_ROOT}/plots/tc/${TC_EVENT_TAG}.stats.json" ]]; then
  mv "${RUN_ROOT}/plots/tc/${TC_EVENT_TAG}.stats.json" "${TC_STATS}"
fi
ln -sfn "data/tc/${TC_EVENT_TAG}.stats.json" "${RUN_ROOT}/tc_normed_pdfs_${TC_EVENT_TAG}_${RUN_ID}_from_predictions.stats.json" 2>/dev/null || true
ln -sfn "plots/tc/${TC_EVENT_TAG}.pdf" "${RUN_ROOT}/tc_normed_pdfs_${TC_EVENT_TAG}_${RUN_ID}_from_predictions.pdf" 2>/dev/null || true

echo "DONE scoreboard_tc_step run_id=${RUN_ID}"
echo "TC_STATS=${TC_STATS}"
echo "TC_PDF=${TC_PDF}"
