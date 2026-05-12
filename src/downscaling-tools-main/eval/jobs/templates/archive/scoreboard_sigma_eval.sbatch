#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
# Canonical sigma-evaluator launcher for scoreboard and manual-eval chains.
#
# Default allocation fits o48->o96 and o96->o320.
# For o320->o1280 and o1280->o2560, submit from an AG host with:
#   sbatch --gpus-per-node=4 --ntasks-per-node=4 scoreboard_sigma_eval.sbatch
# and set NUM_GPUS_PER_MODEL=4 below. Single-GPU AG sigma launches are not
# reliable for those lanes.

#SBATCH --job-name=sigma_eval
#SBATCH --qos=ng
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ID="REPLACE_RUN_ID"          # scoreboard run id (should match eval run root name)
RUN_ROOT=""                      # blank => /home/ecm5702/scratch/eval/<RUN_ID>
CHECKPOINT_REF=""                # checkpoint directory under CKPT_ROOT; blank => RUN_ID
CKPT_NAME="REPLACE_CHECKPOINT_FILENAME.ckpt"   # use non-inference checkpoint (anemoi-*.ckpt)
EXPECTED_STACK_FLAVOR="new"      # new | old
CKPT_ROOT="/home/ecm5702/scratch/aifs/checkpoint"
OUT_CSV=""                       # blank => <RUN_ROOT>/data/sigma_eval.csv
SIGMAS="1,5,10,100"
N_SAMPLES=10
VALIDATION_FREQUENCY="50h"
EXPECTED_LANE="o96_o320"         # o48_o96 | o96_o320 | o320_o1280 | o1280_o2560
NUM_GPUS_PER_MODEL=1              # set to 4 for o320_o1280 / o1280_o2560
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_choice() {
  local value="$1"; shift
  local ok=1
  for c in "$@"; do
    if [[ "${value}" == "${c}" ]]; then
      ok=0
      break
    fi
  done
  if [[ "${ok}" -ne 0 ]]; then
    die "Invalid value '${value}'. Allowed: $*"
  fi
}

[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ "${CKPT_NAME}" != REPLACE_* ]] || die "Set CKPT_NAME."
[[ "${CKPT_NAME}" != inference-* ]] || die \
  "CKPT_NAME='${CKPT_NAME}' looks like an inference checkpoint. Use a non-inference anemoi-*.ckpt for sigma evaluator."
require_choice "${EXPECTED_LANE}" o48_o96 o96_o320 o320_o1280 o1280_o2560
require_choice "${EXPECTED_STACK_FLAVOR}" new old
require_choice "${NUM_GPUS_PER_MODEL}" 1 4

if [[ -z "${CHECKPOINT_REF}" ]]; then
  CHECKPOINT_REF="${RUN_ID}"
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*)
    if [[ "${EXPECTED_STACK_FLAVOR}" == "new" ]]; then
      VENV="/home/ecm5702/dev/.ds-dyn/bin/activate"
    else
      VENV="/home/ecm5702/dev/.ds-old/bin/activate"
    fi
    ;;
  ag*)
    if [[ "${EXPECTED_STACK_FLAVOR}" == "new" ]]; then
      VENV="/home/ecm5702/dev/.ds-ag/bin/activate"
    else
      VENV="/home/ecm5702/dev/.ds-ag-old/bin/activate"
    fi
    ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

if [[ "${EXPECTED_LANE}" == "o320_o1280" || "${EXPECTED_LANE}" == "o1280_o2560" ]]; then
  [[ "${HOST_SHORT}" == ag* ]] || die \
    "Lane ${EXPECTED_LANE} sigma eval must run from AG. Single-GPU fallback is not canonical for this lane."
  [[ "${NUM_GPUS_PER_MODEL}" == "4" ]] || die \
    "Lane ${EXPECTED_LANE} requires NUM_GPUS_PER_MODEL=4. Single-GPU AG sigma launches are not reliable."
  [[ "${SLURM_NTASKS:-1}" == "4" ]] || die \
    "Lane ${EXPECTED_LANE} requires a 4-task allocation. Submit with: sbatch --gpus-per-node=4 --ntasks-per-node=4 scoreboard_sigma_eval.sbatch"
fi

REPO_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
[[ -d "${REPO_ROOT}" ]] || die "Repo root not found: ${REPO_ROOT}"

[[ -f "${VENV}" ]] || die "Missing venv activate script: ${VENV}"
[[ -f "${CKPT_ROOT}/${CHECKPOINT_REF}/${CKPT_NAME}" ]] || die "Missing checkpoint: ${CKPT_ROOT}/${CHECKPOINT_REF}/${CKPT_NAME}"

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="/home/ecm5702/scratch/eval/${RUN_ID}"
fi
LEGACY_SIGMA_DIR="/home/ecm5702/scratch/eval/scoreboards/sigma"
if [[ -z "${OUT_CSV}" ]]; then
  # Canonical output path (eval-data-contract)
  OUT_CSV="${RUN_ROOT}/data/sigma_eval.csv"
fi
mkdir -p "$(dirname "${OUT_CSV}")" "${LEGACY_SIGMA_DIR}"

source "${VENV}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

python -m eval.jobs.checkpoint_profile \
  --name-ckpt "${CHECKPOINT_REF}/${CKPT_NAME}" \
  --ckpt-root "${CKPT_ROOT}" \
  --source-hpc "${HOST_SHORT:0:2}" \
  --host-short "${HOST_SHORT}" \
  --expected-lane "${EXPECTED_LANE}" \
  --expected-stack-flavor "${EXPECTED_STACK_FLAVOR}" \
  --expected-venv "${VENV}"

hostname -s
printf '%s\n' "$VIRTUAL_ENV"
which python
python -V

cd "${REPO_ROOT}"
srun -u \
  --ntasks="${NUM_GPUS_PER_MODEL}" \
  --ntasks-per-node="${NUM_GPUS_PER_MODEL}" \
  python -u -m eval._backends.sigma_evaluator.run_sigma_evaluator \
    --name_exp "${CHECKPOINT_REF}" \
    --name_ckpt "${CKPT_NAME}" \
    --ckpt-root "${CKPT_ROOT}" \
    --sigmas "${SIGMAS}" \
    --n_samples "${N_SAMPLES}" \
    --validation_frequency "${VALIDATION_FREQUENCY}" \
    --out_csv "${OUT_CSV}" \
    --num-gpus-per-model "${NUM_GPUS_PER_MODEL}" \
    --device cuda

# Copy to legacy scoreboard aggregator location
cp "${OUT_CSV}" "${LEGACY_SIGMA_DIR}/${RUN_ID}_sigma_eval.csv" 2>/dev/null || true

echo "DONE: ${OUT_CSV}"
