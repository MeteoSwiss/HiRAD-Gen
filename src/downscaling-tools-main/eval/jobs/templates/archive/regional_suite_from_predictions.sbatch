#!/bin/bash --login
# Canonical regional-suite plotting template for predictions_*.nc files.
#
# Renders a multi-region PDF/PNG suite from one predictions file using:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/region_plotting/plot_regions.py
#
# Preferred launch path for the `o320 -> o1280` manual-eval lane:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
#
# Canonical host/env rule for new-stack O1280 work:
#   - AG login host -> `/home/ecm5702/dev/.ds-ag/bin/activate`
#   - AC login host -> `/home/ecm5702/dev/.ds-dyn/bin/activate`
# Never reuse the env choice from the wrong host family.

#SBATCH --job-name=regional_suite
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
OUT_DIR="${RUN_ROOT}/local_plots_regions_step024"
REGION_NAMES=""             # blank => default suite for the grid
SUITE_KIND="regions"        # regions | storm
SAMPLE_INDEX=0
ENSEMBLE_MEMBER_INDEX=0
MODEL_VARIABLES="x_0,x_interp_0,y_0,y_pred_0,residuals_0,residuals_pred_0"
WEATHER_STATES=""
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
[[ "${SUITE_KIND}" == "regions" || "${SUITE_KIND}" == "storm" ]] || die "SUITE_KIND must be regions or storm (got ${SUITE_KIND})"
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
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] predictions_file=${PREDICTIONS_FILE}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] suite_kind=${SUITE_KIND}"
echo "[INFO] regions=${REGION_NAMES:-default_suite}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

CMD=(
  "${PYTHON}" -m eval._backends.region_plotting.plot_regions
  --predictions-nc "${PREDICTIONS_FILE}"
  --out-dir "${OUT_DIR}"
  --sample-index "${SAMPLE_INDEX}"
  --ensemble-member-index "${ENSEMBLE_MEMBER_INDEX}"
  --model-variables "${MODEL_VARIABLES}"
  --suite-kind "${SUITE_KIND}"
)

if [[ -n "${REGION_NAMES}" ]]; then
  CMD+=(--regions "${REGION_NAMES}")
fi
if [[ -n "${WEATHER_STATES}" ]]; then
  CMD+=(--weather-states "${WEATHER_STATES}")
fi
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
