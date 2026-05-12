#!/bin/bash --login
# Recovery template for walltime-killed prediction runs.
#
# Auto-detects which predictions_YYYYMMDD_stepXXX.nc files are missing
# from an existing run directory and relaunches only the remaining
# date/step combinations.
#
# Common scenario: a full75 run was killed after 12h with 61/75 files
# written. This template picks up the remaining 14 files without
# re-running completed work.
#
# Resource baselines (empirical):
#   Remaining 14/75 files: --time=24:00:00 (conservative)
#   Remaining 4/25 files:  --time=08:00:00
#   O1280 runs are ~2-3x slower than O320 per file.
#
# IMPORTANT: This template targets an existing run directory and passes
# --allow-existing-out-dir plus --allow-overwrite-existing-files.

#SBATCH --job-name=predict_recovery
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
STACK_FLAVOR="new"         # new | old
LANE="o96_o320"            # o96_o320 | o320_o1280 | o1280_o2560
SOURCE_HPC="ac"            # ac | ag

CHECKPOINT_REF="REPLACE_CHECKPOINT_REF"
CKPT_ROOT="/home/ecm5702/scratch/aifs/checkpoint"
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"

# The existing run directory to recover into
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_DIR"

# Full date/step grid — the template will auto-skip completed files
ALL_DATES="20230826,20230827,20230828,20230829,20230830"
ALL_STEPS="24,48,72,96,120"
MEMBERS="1,2,3,4,5,6,7,8,9,10"

DEVICE="cuda"
PRECISION="fp32"
NUM_GPUS_PER_MODEL=1
EXTRA_ARGS_JSON='{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'

# Unsafe mode for bundles without target_hres_* fields
ALLOW_MISSING_TARGET_UNSAFE=0
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_choice() {
  local value="$1"; shift
  local ok=1
  for c in "$@"; do
    if [[ "${value}" == "${c}" ]]; then ok=0; break; fi
  done
  [[ "${ok}" -eq 0 ]] || die "Invalid value '${value}'. Allowed: $*"
}

require_choice "${STACK_FLAVOR}" new old
require_choice "${LANE}" o96_o320 o320_o1280 o1280_o2560
require_choice "${SOURCE_HPC}" ac ag
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${INPUT_ROOT}" ]] || die "INPUT_ROOT does not exist: ${INPUT_ROOT}"

PREDICTIONS_DIR="${RUN_ROOT}/predictions"
mkdir -p "${PREDICTIONS_DIR}" "${RUN_ROOT}/logs"

# --- Auto-detect missing files ---
echo "=== Scanning for missing prediction files ==="

MISSING_DATES=()
MISSING_COMBOS=()
EXISTING=0
MISSING=0

IFS=',' read -ra DATES_ARR <<< "${ALL_DATES}"
IFS=',' read -ra STEPS_ARR <<< "${ALL_STEPS}"

for d in "${DATES_ARR[@]}"; do
  date_has_missing=0
  for s in "${STEPS_ARR[@]}"; do
    step_padded=$(printf "%03d" "${s}")
    fname="predictions_${d}_step${step_padded}.nc"
    # Also check unpadded variant
    fname_unpadded="predictions_${d}_step${s}.nc"
    if [[ -f "${PREDICTIONS_DIR}/${fname}" ]] || [[ -f "${PREDICTIONS_DIR}/${fname_unpadded}" ]]; then
      EXISTING=$((EXISTING + 1))
    else
      MISSING=$((MISSING + 1))
      MISSING_COMBOS+=("${d}/step${s}")
      date_has_missing=1
    fi
  done
  if [[ "${date_has_missing}" -eq 1 ]]; then
    MISSING_DATES+=("${d}")
  fi
done

echo "[INFO] Existing: ${EXISTING}, Missing: ${MISSING}"

if [[ "${MISSING}" -eq 0 ]]; then
  echo "[INFO] All prediction files already exist. Nothing to do."
  exit 0
fi

echo "[INFO] Missing files:"
for combo in "${MISSING_COMBOS[@]}"; do
  echo "  ${combo}"
done

# Build the dates CSV for only dates that have missing files
RECOVERY_DATES=$(IFS=','; echo "${MISSING_DATES[*]}")
echo "[INFO] Recovery dates: ${RECOVERY_DATES}"

# --- Resolve environment ---
HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *)   die "Unknown cluster: ${HOST_SHORT}" ;;
esac

case "${CLUSTER}_${STACK_FLAVOR}" in
  ac_new) source /home/ecm5702/dev/.ds-dyn/bin/activate ;;
  ag_new) source /home/ecm5702/dev/.ds-ag/bin/activate ;;
  ac_old) source /home/ecm5702/dev/.ds-old/bin/activate ;;
  ag_old) source /home/ecm5702/dev/.ds-ag-old/bin/activate ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PYTHON="$(which python)"

echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] python=${PYTHON} ($(${PYTHON} -V 2>&1))"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"

[[ -d "${CKPT_ROOT}" ]] || die "CKPT_ROOT does not exist: ${CKPT_ROOT}"

# --- Build and run recovery command ---
cd "${REPO_ROOT}"

CMD=(
  "${PYTHON}" eval/jobs/generate_predictions_25_files.py
  --input-root "${INPUT_ROOT}"
  --out-dir "${PREDICTIONS_DIR}"
  --dates "${RECOVERY_DATES}"
  --steps "${ALL_STEPS}"
  --members "${MEMBERS}"
  --device "${DEVICE}"
  --precision "${PRECISION}"
  --allow-existing-out-dir
  --allow-overwrite-existing-files
)

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  CMD+=(--name-ckpt "${CHECKPOINT_REF}" --ckpt-root "${CKPT_ROOT}")
  CMD+=(--num-gpus-per-model "${NUM_GPUS_PER_MODEL}")
else
  CMD+=(--ckpt-id "${CHECKPOINT_REF}" --ckpt-root "${CKPT_ROOT}")
fi

if [[ "${ALLOW_MISSING_TARGET_UNSAFE}" -eq 1 ]]; then
  CMD+=(--allow-missing-target-unsafe)
fi

if [[ -n "${EXTRA_ARGS_JSON}" ]]; then
  CMD+=(--extra-args-json "${EXTRA_ARGS_JSON}")
fi

echo "[INFO] Running recovery: ${CMD[*]}"
"${CMD[@]}"

# --- Verify ---
echo ""
echo "=== Post-recovery verification ==="
FINAL_COUNT=$(find "${PREDICTIONS_DIR}" -maxdepth 1 -name 'predictions_*.nc' -type f | wc -l)
TOTAL_EXPECTED=$(( ${#DATES_ARR[@]} * ${#STEPS_ARR[@]} ))
echo "[INFO] Final file count: ${FINAL_COUNT} / ${TOTAL_EXPECTED}"

if [[ "${FINAL_COUNT}" -lt "${TOTAL_EXPECTED}" ]]; then
  echo "[WARN] Still missing $(( TOTAL_EXPECTED - FINAL_COUNT )) files. May need another recovery run."
else
  echo "[INFO] All prediction files present."
fi

echo "DONE: ${PREDICTIONS_DIR}"
