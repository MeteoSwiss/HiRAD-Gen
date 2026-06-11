#!/bin/bash --login
# Canonical TC contour-suite plotting template for predictions_*.nc files.
#
# Renders contour-style storm-centered plots from one predictions file using:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/region_plotting/plot_tc_contours_from_predictions.py
#
# This is a prediction-only CPU plotting path: no generation and no intermediate-state rebuild.
#
# Preferred launch path for the `o320 -> o1280` manual-eval lane:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
#
# Canonical host/env rule for new-stack O1280 work:
#   - AG login host -> `/home/ecm5702/dev/.ds-ag/bin/activate`
#   - AC login host -> `/home/ecm5702/dev/.ds-dyn/bin/activate`
# Never reuse the env choice from the wrong host family.

#SBATCH --job-name=tc_contours
#SBATCH --qos=ng
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_FILE="${RUN_ROOT}/predictions/predictions_20230829_step024.nc"
OUT_DIR="${RUN_ROOT}/tc_contour_plots_step024"
REGION_NAMES="idalia,franklin"
SAMPLE_INDEX=0
ENSEMBLE_MEMBER_INDEX=0
ALSO_PNG=1
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"


die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *) die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -f "${PREDICTIONS_FILE}" ]] || die "Predictions file does not exist: ${PREDICTIONS_FILE}"

case "${CLUSTER}" in
  ac)
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag)
    source /home/ecm5702/dev/.ds-ag/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
esac
[[ -x "${PYTHON}" ]] || die "Expected host-matched Python not found: ${PYTHON}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] predictions_file=${PREDICTIONS_FILE}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] regions=${REGION_NAMES}"

CMD=(
  "${PYTHON}" -m eval._backends.region_plotting.plot_tc_contours_from_predictions
  --predictions-nc "${PREDICTIONS_FILE}"
  --out-dir "${OUT_DIR}"
  --regions "${REGION_NAMES}"
  --sample-index "${SAMPLE_INDEX}"
  --ensemble-member-index "${ENSEMBLE_MEMBER_INDEX}"
)

if [[ "${ALSO_PNG}" == "1" ]]; then
  CMD+=(--also-png)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

PDF_COUNT="$(find "${OUT_DIR}" -type f -name '*.pdf' | wc -l | tr -d ' ')"
PNG_COUNT="$(find "${OUT_DIR}" -type f -name '*.png' | wc -l | tr -d ' ')"
[[ "${PDF_COUNT}" -gt 0 ]] || die "No PDF outputs found under ${OUT_DIR}"
if [[ "${ALSO_PNG}" == "1" ]]; then
  [[ "${PNG_COUNT}" -gt 0 ]] || die "No PNG outputs found under ${OUT_DIR}"
fi

echo "DONE: ${OUT_DIR} (pdf=${PDF_COUNT} png=${PNG_COUNT})"
