#!/bin/bash --login
# TC member-map plotting template — renders per-member 2x3 spatial grids
# (MSLP/Wind × Input/Prediction/Target) from predictions_*.nc files using:
#   eval._backends.tc.workflows member-maps
#
# One sbatch per lane. Each produces a single multi-page PDF per TC event.
#
# Canonical launch path: submit_o320_o1280_manual_eval_flow.sh (or o48/scoreboard).

#SBATCH --job-name=tc_member_maps
#SBATCH --qos=ng
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=0
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"
OUT_DIR="${RUN_ROOT}/tc_member_maps"
EVENTS="idalia,franklin,franklin_idalia"
DATE="20230828"
STEPS="48"
MEMBERS="0,1,2,3,4"
RUN_LABEL="REPLACE_RUN_LABEL"
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
[[ -d "${PREDICTIONS_DIR}" ]] || die "PREDICTIONS_DIR does not exist: ${PREDICTIONS_DIR}"

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
echo "[INFO] predictions_dir=${PREDICTIONS_DIR}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] events=${EVENTS}"
echo "[INFO] date=${DATE} steps=${STEPS} members=${MEMBERS}"
echo "[INFO] run_label=${RUN_LABEL}"

CMD=(
  "${PYTHON}" -m eval._backends.tc.workflows member-maps
  --predictions-dir "${PREDICTIONS_DIR}"
  --run-label "${RUN_LABEL}"
  --date "${DATE}"
  --steps "${STEPS}"
  --members "${MEMBERS}"
  --events "${EVENTS}"
  --outdir "${OUT_DIR}"
)

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

PDF_COUNT="$(find "${OUT_DIR}" -type f -name '*.pdf' | wc -l | tr -d ' ')"
[[ "${PDF_COUNT}" -gt 0 ]] || die "No PDF outputs found under ${OUT_DIR}"

echo "DONE: ${OUT_DIR} (pdf=${PDF_COUNT})"
