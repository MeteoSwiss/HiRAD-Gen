#!/bin/bash --login
#SBATCH --job-name=manual_predict_one_bundle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --qos=ng
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
STACK_FLAVOR="new"         # new | old
LANE="o96_o320"            # o48_o96 | o96_o320 | o320_o1280 | o1280_o2560
SOURCE_HPC="ac"            # ac | ag | leonardo | jupiter

CHECKPOINT_REF="REPLACE_CHECKPOINT_REF"
CHECKPOINT_SHORT="REPLACE_CHECKPOINT_SHORT"
CKPT_ROOT="/home/ecm5702/scratch/aifs/checkpoint"
BUNDLE_NC="REPLACE_INPUT_BUNDLE_NC"

OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_DATE_UTC="$(date -u +%Y%m%d)"
RUN_SUFFIX=""              # optional, safe chars only

DEVICE="cuda"
PRECISION="fp32"
VALIDATION_FREQUENCY="50h"
EXTRA_ARGS_JSON='{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'

# Old stack currently allows missing target in code path; set to 1 to enforce y anyway.
REQUIRE_Y_ON_OLD_STACK=0
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_extra_args_json() {
  local raw="$1"
  python - "${raw}" <<'PYJSON'
import json
import sys

raw = sys.argv[1]
try:
    parsed = json.loads(raw) if raw else {}
except json.JSONDecodeError as exc:
    raise SystemExit(
        "EXTRA_ARGS_JSON is not valid JSON. Keep the whole value shell-quoted, for example \"EXTRA_ARGS_JSON='\"{...}\"'\". "
        f"Original error: {exc}"
    )
if not isinstance(parsed, dict):
    raise SystemExit(f"EXTRA_ARGS_JSON must decode to a JSON object, got {type(parsed).__name__}")
print(json.dumps(parsed, separators=(",", ":")))
PYJSON
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

require_choice "${STACK_FLAVOR}" new old
require_choice "${LANE}" o48_o96 o96_o320 o320_o1280 o1280_o2560
require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ "${BUNDLE_NC}" != REPLACE_* ]] || die "Set BUNDLE_NC."
[[ -f "${BUNDLE_NC}" ]] || die "Bundle does not exist: ${BUNDLE_NC}"
[[ "${BUNDLE_NC}" == *"/${LANE}/"* ]] || die "Bundle path does not match lane ${LANE}: ${BUNDLE_NC}"
if [[ "${LANE}" == "o1280_o2560" ]]; then
  die "strict_manual_predict_one_bundle.sbatch is not a valid proof route for LANE=o1280_o2560. The imported O2560 checkpoint family requires 4 GPU ranks. Use /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o1280_o2560_manual_eval_flow.sh instead."
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
REPO_ROOT="${PROJECT_ROOT}"

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  CODE_ROOT="${REPO_ROOT}"
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

[[ -d "${REPO_ROOT}" ]] || die "Project root not found: ${REPO_ROOT}"
[[ -d "${CODE_ROOT}" ]] || die "Code root not found: ${CODE_ROOT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"

RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
[[ ! -e "${RUN_DIR}" ]] || die "Run directory already exists: ${RUN_DIR}"
mkdir -p "${RUN_DIR}/logs"

source "${VENV_ACTIVATE}"
BASE_PYTHONPATH="${PYTHONPATH:-}"
NORMALIZED_EXTRA_ARGS_JSON="$(normalize_extra_args_json "${EXTRA_ARGS_JSON}")"
EXTRA_ARGS_JSON="${NORMALIZED_EXTRA_ARGS_JSON}"
if [[ -n "${BASE_PYTHONPATH}" ]]; then
  export PYTHONPATH="${CODE_ROOT}:${BASE_PYTHONPATH}"
  CHECKPOINT_PROFILE_PYTHONPATH="${REPO_ROOT}:${BASE_PYTHONPATH}"
else
  export PYTHONPATH="${CODE_ROOT}"
  CHECKPOINT_PROFILE_PYTHONPATH="${REPO_ROOT}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction

PYTHONPATH="${CHECKPOINT_PROFILE_PYTHONPATH}" \
python -m eval.jobs.checkpoint_profile \
  --name-ckpt "${CHECKPOINT_REF}" \
  --ckpt-root "${CKPT_ROOT}" \
  --source-hpc "${SOURCE_HPC}" \
  --host-short "${HOST_SHORT}" \
  --expected-lane "${LANE}" \
  --expected-stack-flavor "${STACK_FLAVOR}" \
  --expected-venv "${VENV_ACTIVATE}"

{
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host_short=${HOST_SHORT}"
  echo "host_family=${HOST_FAMILY}"
  echo "source_hpc=${SOURCE_HPC}"
  echo "stack_flavor=${STACK_FLAVOR}"
  echo "lane=${LANE}"
  echo "venv_activate=${VENV_ACTIVATE}"
  echo "python=$(which python)"
  echo "python_version=$(python -V 2>&1)"
  echo "code_root=${CODE_ROOT}"
  echo "checkpoint_ref=${CHECKPOINT_REF}"
  echo "ckpt_root=${CKPT_ROOT}"
  echo "bundle_nc=${BUNDLE_NC}"
  echo "output_predictions_nc=${RUN_DIR}/predictions.nc"
  echo "device=${DEVICE}"
  echo "precision=${PRECISION}"
  echo "validation_frequency=${VALIDATION_FREQUENCY}"
  echo "extra_args_json=${EXTRA_ARGS_JSON}"
} > "${RUN_DIR}/manual_inference_run_info.txt"

python - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${RUN_ID}" "${RUN_DIR}" "${STACK_FLAVOR}" "${LANE}" "${SOURCE_HPC}" "${HOST_SHORT}" "${VENV_ACTIVATE}" "${CODE_ROOT}" "${CHECKPOINT_REF}" "${CKPT_ROOT}" "${BUNDLE_NC}" "${EXTRA_ARGS_JSON}" <<'PY'
import json
import re
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
run_id = sys.argv[2]
run_dir = sys.argv[3]
stack_flavor = sys.argv[4]
lane = sys.argv[5]
source_hpc = sys.argv[6]
host_short = sys.argv[7]
venv_activate = sys.argv[8]
code_root = sys.argv[9]
checkpoint_ref = sys.argv[10]
ckpt_root = sys.argv[11]
bundle_nc = sys.argv[12]
sampling_config_json = sys.argv[13]

date_match = re.search(r"(20\d{6})", Path(bundle_nc).name)
step_match = re.search(r"step0*([0-9]{1,3})", Path(bundle_nc).name)
checkpoint_token = checkpoint_ref.split("/", 1)[0]
checkpoint_path = Path(checkpoint_ref)
if not checkpoint_path.is_absolute():
    checkpoint_path = Path(ckpt_root) / checkpoint_ref

config = {
    "run_id": run_id,
    "artifacts_root": run_dir,
    "stack_flavor": stack_flavor,
    "lane": lane,
    "source_hpc": source_hpc,
    "host_short": host_short,
    "venv_activate": venv_activate,
    "code_root": code_root,
    "sampling_config_json": sampling_config_json,
    "checkpoint": {
        "run_id": checkpoint_token,
        "path": str(checkpoint_path),
    },
    "source": {
        "bundle_nc": bundle_nc,
        "bundle_scope": {
            "dates": [int(date_match.group(1))] if date_match else [],
            "members": [],
            "steps_hours": [int(step_match.group(1))] if step_match else [],
        },
    },
}
try:
    config["model"] = {"development_hacks": {"extra_args": json.loads(sampling_config_json)}}
except json.JSONDecodeError:
    pass

out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

cd "${CODE_ROOT}"

python -m manual_inference.prediction.predict from-bundle \
  --name-ckpt "${CHECKPOINT_REF}" \
  --ckpt-root "${CKPT_ROOT}" \
  --device "${DEVICE}" \
  --bundle-nc "${BUNDLE_NC}" \
  --validation-frequency "${VALIDATION_FREQUENCY}" \
  --extra-args-json "${EXTRA_ARGS_JSON}" \
  --precision "${PRECISION}" \
  --allow-existing-output-dir \
  --out "${RUN_DIR}/predictions.nc"

PRED_NC="${RUN_DIR}/predictions.nc"
[[ -f "${PRED_NC}" ]] || die "Expected output file missing: ${PRED_NC}"

python - "${PRED_NC}" "${STACK_FLAVOR}" "${REQUIRE_Y_ON_OLD_STACK}" <<'PY'
import sys
import xarray as xr

path = sys.argv[1]
stack = sys.argv[2]
require_old = int(sys.argv[3])
with xr.open_dataset(path) as ds:
    required = ["x", "y_pred"]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise SystemExit(f"Missing required output variables {missing} in {path}")
    if stack == "new" or require_old == 1:
        if "y" not in ds.variables:
            raise SystemExit(
                f"Missing y in {path}. For new stack this is a hard requirement."
            )
print(f"Output validation OK: {path}")
PY

echo "DONE: ${PRED_NC}"
