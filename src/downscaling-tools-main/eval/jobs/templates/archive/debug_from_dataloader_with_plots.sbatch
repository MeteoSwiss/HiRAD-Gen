#!/bin/bash --login
# Canonical combined template: debug predict (from-dataloader) + immediate local plots.
#
# Runs `manual_inference.prediction.predict from-dataloader` with --debug-from-dataloader
# on a multi-GPU node, validates the output, then immediately renders local region plots
# via `eval._backends.region_plotting.plot_one_date_local`.
#
# Designed for the o1280_o2560 lane (256 G model, 4-GPU DDP required) but accepts
# o48_o96, o96_o320, and o320_o1280 with NTASKS=1 and no EXTRA_PREDICT_FLAGS.
#
# For predict-only (no plots) see:
#   codex_login_node_templates/strict_debug_from_dataloader.sbatch
# For plots-only from an existing predictions tree see:
#   eval/jobs/templates/local_plots_one_date_from_predictions.sbatch

#SBATCH --job-name=dl_debug_plot
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
STACK_FLAVOR="REPLACE_STACK_FLAVOR"     # new | old
LANE="REPLACE_LANE"                      # o48_o96 | o96_o320 | o320_o1280 | o1280_o2560
SOURCE_HPC="REPLACE_SOURCE_HPC"         # ac | ag | leonardo | jupiter

# Full training checkpoint — used for the predict step.
CHECKPOINT_REF="REPLACE_CHECKPOINT_REF"
CHECKPOINT_SHORT="REPLACE_CHECKPOINT_SHORT"
CKPT_ROOT="/home/ecm5702/scratch/aifs/checkpoint"

# Inference companion checkpoint — used for the profiler when USE_INFERENCE_COMPANION=true.
# Set USE_INFERENCE_COMPANION=true when the full training ckpt is too large for the
# login-node memory supervisor (typically required for o1280_o2560).
USE_INFERENCE_COMPANION=false
INFERENCE_COMPANION="REPLACE_INFERENCE_COMPANION"

OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_DATE_UTC="$(date -u +%Y%m%d)"
RUN_SUFFIX="REPLACE_RUN_SUFFIX"         # e.g. "dataloader_plot_v1"

IDX=0
N_SAMPLES=1
MEMBERS="0"
VALIDATION_FREQUENCY="50h"
EXTRA_ARGS_JSON='{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'

# Number of GPU ranks for srun DDP. Use 4 for o1280_o2560; 1 for o48_o96/o96_o320/o320_o1280.
NTASKS=4

# Lane-specific extra predict flags. Leave empty for o48_o96 / o96_o320 / o320_o1280.
# For o1280_o2560, use the maintained contract from submit_o1280_o2560_manual_eval_flow.sh:
#   EXTRA_PREDICT_FLAGS="--output-weather-state-mode all --output-weather-states 10u,10v,2t,msl --slim-output"
EXTRA_PREDICT_FLAGS=""

# Variables to plot; passed to --weather-states. e.g. "2t,10u,10v,msl"
PLOT_WEATHER_STATES="REPLACE_PLOT_WEATHER_STATES"

# Required acknowledgement string for debug-mode template usage.
# Set to exactly: I_UNDERSTAND_THIS_IS_DEBUG_ONLY
DEBUG_ACK="REPLACE_DEBUG_ACK"
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
require_choice "${LANE}" o48_o96 o96_o320 o320_o1280 o1280_o2560
require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ "${RUN_SUFFIX}" != REPLACE_* ]] || die "Set RUN_SUFFIX."
[[ "${PLOT_WEATHER_STATES}" != REPLACE_* ]] || die "Set PLOT_WEATHER_STATES."
[[ "${DEBUG_ACK}" == "I_UNDERSTAND_THIS_IS_DEBUG_ONLY" ]] || die "Set DEBUG_ACK to the required acknowledgement string."
if "${USE_INFERENCE_COMPANION}"; then
  [[ "${INFERENCE_COMPANION}" != REPLACE_* ]] || die "USE_INFERENCE_COMPANION=true requires INFERENCE_COMPANION to be set."
fi
if [[ "${LANE}" == "o1280_o2560" && "${NTASKS}" -lt 4 ]]; then
  die "LANE=o1280_o2560 requires NTASKS>=4 for the debug dataloader fallback."
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  CODE_ROOT="/home/ecm5702/dev/downscaling-tools"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-dyn/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag/bin/activate"
  fi
else
  CODE_ROOT="/home/ecm5702/dev/anemoi_2024/downscaling-tools-OLD"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-old/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag-old/bin/activate"
  fi
fi

[[ -d "${CODE_ROOT}" ]] || die "Code root not found: ${CODE_ROOT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"

RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}_${RUN_SUFFIX}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || die "Run directory already exists: ${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"

PRED_NC="${RUN_DIR}/predictions.nc"
PLOT_DIR="${RUN_DIR}/local_plots_from_dataloader"
mkdir -p "${PLOT_DIR}"

source "${VENV_ACTIVATE}"
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

echo "[INFO] host=${HOST_SHORT} family=${HOST_FAMILY}"
echo "[INFO] run_dir=${RUN_DIR}"
echo "[INFO] python=$(which python) ($(python -V 2>&1))"
echo "[INFO] stack_flavor=${STACK_FLAVOR} lane=${LANE}"

# Checkpoint profiler
if "${USE_INFERENCE_COMPANION}"; then
  PROFILER_CKPT="${INFERENCE_COMPANION}"
  PROFILER_FLAGS=(--allow-inference-companion)
else
  PROFILER_CKPT="${CHECKPOINT_REF}"
  PROFILER_FLAGS=()
fi

python -m eval.jobs.checkpoint_profile \
  --name-ckpt "${PROFILER_CKPT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --source-hpc "${SOURCE_HPC}" \
  --host-short "${HOST_SHORT}" \
  --expected-lane "${LANE}" \
  --expected-stack-flavor "${STACK_FLAVOR}" \
  --expected-venv "${VENV_ACTIVATE}" \
  "${PROFILER_FLAGS[@]}"

# Provenance record
{
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host_short=${HOST_SHORT}"
  echo "host_family=${HOST_FAMILY}"
  echo "source_hpc=${SOURCE_HPC}"
  echo "stack_flavor=${STACK_FLAVOR}"
  echo "lane=${LANE}"
  echo "debug_ack=${DEBUG_ACK}"
  echo "venv_activate=${VENV_ACTIVATE}"
  echo "python=$(which python)"
  echo "python_version=$(python -V 2>&1)"
  echo "code_root=${CODE_ROOT}"
  echo "checkpoint_ref=${CHECKPOINT_REF}"
  echo "inference_companion=${INFERENCE_COMPANION:-none}"
  echo "use_inference_companion=${USE_INFERENCE_COMPANION}"
  echo "ckpt_root=${CKPT_ROOT}"
  echo "idx=${IDX}"
  echo "n_samples=${N_SAMPLES}"
  echo "members=${MEMBERS}"
  echo "validation_frequency=${VALIDATION_FREQUENCY}"
  echo "ntasks=${NTASKS}"
  echo "extra_args_json=${EXTRA_ARGS_JSON}"
  echo "extra_predict_flags=${EXTRA_PREDICT_FLAGS[*]:-}"
  echo "plot_weather_states=${PLOT_WEATHER_STATES}"
} > "${RUN_DIR}/manual_inference_run_info.txt"

python - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${RUN_ID}" "${RUN_DIR}" "${STACK_FLAVOR}" "${LANE}" "${SOURCE_HPC}" "${HOST_SHORT}" "${VENV_ACTIVATE}" "${CODE_ROOT}" "${CHECKPOINT_REF}" "${INFERENCE_COMPANION}" "${CKPT_ROOT}" "${IDX}" "${N_SAMPLES}" "${MEMBERS}" "${VALIDATION_FREQUENCY}" "${NTASKS}" "${EXTRA_ARGS_JSON}" "${PLOT_WEATHER_STATES}" "${DEBUG_ACK}" <<'PY'
import json
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
config = {
    "run_id": sys.argv[2],
    "artifacts_root": sys.argv[3],
    "stack_flavor": sys.argv[4],
    "lane": sys.argv[5],
    "source_hpc": sys.argv[6],
    "host_short": sys.argv[7],
    "venv_activate": sys.argv[8],
    "code_root": sys.argv[9],
    "checkpoint": {"path": sys.argv[10]},
    "inference_companion": sys.argv[11],
    "source": {
        "mode": "debug_from_dataloader",
        "ckpt_root": sys.argv[12],
        "idx": int(sys.argv[13]),
        "n_samples": int(sys.argv[14]),
        "members": [int(token.strip()) for token in sys.argv[15].split(",") if token.strip()],
        "validation_frequency": sys.argv[16],
    },
    "debug": {
        "ntasks": int(sys.argv[17]),
        "extra_args_json": sys.argv[18],
        "plot_weather_states": sys.argv[19],
        "acknowledgement": sys.argv[20],
        "proof_only": True,
    },
}
out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

cd "${CODE_ROOT}"

# Build --debug-from-dataloader flag (new stack only)
NEW_DEBUG_FLAG=()
if [[ "${STACK_FLAVOR}" == "new" ]]; then
  NEW_DEBUG_FLAG=(--debug-from-dataloader)
fi
EXTRA_PREDICT_ARGS=()
if [[ -n "${EXTRA_PREDICT_FLAGS}" ]]; then
  read -r -a EXTRA_PREDICT_ARGS <<< "${EXTRA_PREDICT_FLAGS}"
fi

echo "[INFO] starting distributed dataloader prediction (ntasks=${NTASKS})"

srun --ntasks="${NTASKS}" --ntasks-per-node="${NTASKS}" \
  python -m manual_inference.prediction.predict from-dataloader \
  --name-ckpt "${CHECKPOINT_REF}" \
  --ckpt-root "${CKPT_ROOT}" \
  --device cuda \
  --idx "${IDX}" \
  --n-samples "${N_SAMPLES}" \
  --members "${MEMBERS}" \
  --validation-frequency "${VALIDATION_FREQUENCY}" \
  --extra-args-json "${EXTRA_ARGS_JSON}" \
  "${NEW_DEBUG_FLAG[@]}" \
  --allow-existing-output-dir \
  "${EXTRA_PREDICT_ARGS[@]}" \
  --out "${PRED_NC}"

[[ -f "${PRED_NC}" ]] || die "Expected output file missing: ${PRED_NC}"

echo "[INFO] prediction complete: ${PRED_NC}"

# Quick xarray validation
python - "${PRED_NC}" <<'PY'
import sys
import xarray as xr
path = sys.argv[1]
with xr.open_dataset(path) as ds:
    required = ["y_pred", "y"]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise SystemExit(f"Missing required variables {missing} in {path}")
print(f"Output validation OK: {path}")
PY

echo "[INFO] rendering local plots"
python -m eval._backends.region_plotting.plot_one_date_local \
  --predictions-nc "${PRED_NC}" \
  --out "${PLOT_DIR}/member00_baseline.pdf" \
  --also-png "${PLOT_DIR}/member00_baseline.png" \
  --run-label "${RUN_ID}" \
  --weather-states "${PLOT_WEATHER_STATES}"

echo "[INFO] done"
find "${RUN_DIR}" -maxdepth 2 -type f | sort
