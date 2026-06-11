#!/bin/bash --login
# Canonical batch manual-inference template for multi-bundle prediction runs.
# For `LANE=o320_o1280`, prefer the repo-owned login-node helper:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
# It renders this template with the tested 4-GPU / 32-CPU / 24h posture and
# wires local plots, spectra, and TC follow-ups without ad hoc script forks.
# For `LANE=o1280_o2560`, prefer the repo-owned login-node helper:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o1280_o2560_manual_eval_flow.sh
# That helper enforces the current surface-only DestinE contract and the
# required 4-GPU predict posture for the imported checkpoint family.
#SBATCH --job-name=manual_predict_x_bundle
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --mem=0
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
INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia"

# Date policy:
# - DATE_PRESET=default: 20230826..20230830 (default behavior)
# - DATE_PRESET=aug16_30: 20230816..20230830 (explicitly asked behavior)
# - DATES=<csv> overrides DATE_PRESET completely.
DATE_PRESET="default"      # default | aug16_30
DATES=""                   # explicit CSV override (example: 20230816,20230817)
STEPS="24,48,72,96,120"
# Optional exact date:step list. When set, it overrides the cartesian product
# of DATES × STEPS for prediction-file selection.
# Example: 20230828:24,20230829:48,20230830:72
BUNDLE_PAIRS=""
MEMBERS="1,2,3,4,5,6,7,8,9,10"

OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_DATE_UTC="$(date -u +%Y%m%d)"
RUN_SUFFIX=""              # optional, safe chars only
RUN_ID_OVERRIDE=""         # optional full run-id override

DEVICE="cuda"
PRECISION="fp32"
VALIDATION_FREQUENCY="50h"  # used by new stack only
NUM_GPUS_PER_MODEL=1        # used by new stack only; o320_o1280 should use 4
OUTPUT_WEATHER_STATE_MODE="surface-plus-core-pl"
OUTPUT_WEATHER_STATES=""
SLIM_OUTPUT=1
# Keep the whole JSON blob shell-quoted as one string; outer single quotes are safest.
EXTRA_ARGS_JSON='{"schedule_type":"experimental_piecewise","num_steps":30,"sigma_max":1000.0,"sigma_transition":10.0,"sigma_min":0.03,"high_schedule_type":"exponential","low_schedule_type":"karras","num_steps_high":10,"num_steps_low":20,"rho":7.0,"sampler":"heun","S_churn":2.5,"S_min":0.75,"S_max":1000.0,"S_noise":1.05}'

# MLflow experiment directory for enriching EXPERIMENT_CONFIG with training params
MLFLOW_EXPERIMENT_DIR=""

# Old stack currently allows missing target in code path; keep default strict.
ALLOW_MISSING_TARGET_OLD=0
# Keep this at 0 unless you intentionally overwrite a previous run.
ALLOW_EXISTING_RUN_DIR=0
ALLOW_REBUILT_BUNDLE_ROOT=0
# Keep these at 0 unless you intentionally reuse an existing predictions/ directory.
ALLOW_EXISTING_PREDICTIONS_DIR=0
ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES=0
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_extra_args_json() {
  local raw="$1"
  python3 - "${raw}" <<'PYJSON'
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

normalize_bundle_pairs() {
  local raw="$1"
  python3 - "${raw}" <<'PYBUNDLE'
import sys

raw = sys.argv[1].strip()
if not raw:
    print("")
    raise SystemExit(0)

seen = set()
normalized = []
for token in raw.split(","):
    token = token.strip()
    if not token:
        continue
    if ":" not in token:
        raise SystemExit(
            "BUNDLE_PAIRS entries must use date:step format, for example 20230828:24."
        )
    date_raw, step_raw = token.split(":", 1)
    date_raw = date_raw.strip()
    step_raw = step_raw.strip()
    if len(date_raw) != 8 or not date_raw.isdigit():
        raise SystemExit(
            f"Invalid BUNDLE_PAIRS date {date_raw!r}; expected YYYYMMDD."
        )
    if not step_raw.isdigit():
        raise SystemExit(
            f"Invalid BUNDLE_PAIRS step {step_raw!r}; expected integer hours."
        )
    key = f"{date_raw}:{int(step_raw)}"
    if key in seen:
        continue
    seen.add(key)
    normalized.append(key)
print(",".join(normalized))
PYBUNDLE
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
require_choice "${DATE_PRESET}" default aug16_30
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ -d "${INPUT_ROOT}" ]] || die "INPUT_ROOT does not exist: ${INPUT_ROOT}"
[[ "${ALLOW_EXISTING_RUN_DIR}" == "0" || "${ALLOW_EXISTING_RUN_DIR}" == "1" ]] || die "ALLOW_EXISTING_RUN_DIR must be 0 or 1."
[[ "${ALLOW_REBUILT_BUNDLE_ROOT}" == "0" || "${ALLOW_REBUILT_BUNDLE_ROOT}" == "1" ]] || die "ALLOW_REBUILT_BUNDLE_ROOT must be 0 or 1."
[[ "${ALLOW_EXISTING_PREDICTIONS_DIR}" == "0" || "${ALLOW_EXISTING_PREDICTIONS_DIR}" == "1" ]] || die "ALLOW_EXISTING_PREDICTIONS_DIR must be 0 or 1."
[[ "${ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES}" == "0" || "${ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES}" == "1" ]] || die "ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES must be 0 or 1."
[[ "${SLIM_OUTPUT}" == "0" || "${SLIM_OUTPUT}" == "1" ]] || die "SLIM_OUTPUT must be 0 or 1."
if [[ -n "${RUN_ID_OVERRIDE}" && ! "${RUN_ID_OVERRIDE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  die "Unsafe RUN_ID_OVERRIDE: ${RUN_ID_OVERRIDE}"
fi

if [[ "${INPUT_ROOT}" == *"/${LANE}/"* ]]; then
  INPUT_ROOT_MODE="lane_canonical_root"
elif [[ "${ALLOW_REBUILT_BUNDLE_ROOT}" -eq 1 ]]; then
  case "$(basename "${INPUT_ROOT}")" in
    bundles_with_y*)
      INPUT_ROOT_MODE="rebuilt_truth_bundle_root"
      ;;
    *)
      die "ALLOW_REBUILT_BUNDLE_ROOT=1 requires INPUT_ROOT basename like bundles_with_y*: ${INPUT_ROOT}"
      ;;
  esac
else
  die "INPUT_ROOT does not match lane ${LANE}: ${INPUT_ROOT}. For rebuilt truth-aware bundle roots, set ALLOW_REBUILT_BUNDLE_ROOT=1."
fi

if [[ -z "${DATES}" ]]; then
  if [[ "${DATE_PRESET}" == "default" ]]; then
    DATES="20230826,20230827,20230828,20230829,20230830"
  else
    DATES="20230816,20230817,20230818,20230819,20230820,20230821,20230822,20230823,20230824,20230825,20230826,20230827,20230828,20230829,20230830"
  fi
fi

BUNDLE_PAIRS="$(normalize_bundle_pairs "${BUNDLE_PAIRS}")"

if [[ "${LANE}" == "o320_o1280" && "${NUM_GPUS_PER_MODEL}" -lt 4 ]]; then
  die "LANE=o320_o1280 requires NUM_GPUS_PER_MODEL>=4. Use /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh or submit this template with matching 4-GPU scheduler overrides."
fi
if [[ "${LANE}" == "o1280_o2560" ]]; then
  if [[ "${NUM_GPUS_PER_MODEL}" -lt 4 ]]; then
    die "LANE=o1280_o2560 requires NUM_GPUS_PER_MODEL>=4. Use /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o1280_o2560_manual_eval_flow.sh or submit this template with matching 4-GPU scheduler overrides."
  fi
  if [[ -z "${OUTPUT_WEATHER_STATES}" ]]; then
    die "LANE=o1280_o2560 requires explicit OUTPUT_WEATHER_STATES (current contract: 10u,10v,2t,msl)."
  fi
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

REPO_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

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

[[ -d "${CODE_ROOT}" ]] || die "Code root not found: ${CODE_ROOT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"
CHECKPOINT_PROFILE_ROOT="${REPO_ROOT}"
[[ -d "${CHECKPOINT_PROFILE_ROOT}" ]] || die "Checkpoint profile root not found: ${CHECKPOINT_PROFILE_ROOT}"

if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE}"
else
  RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
  if [[ -n "${RUN_SUFFIX}" ]]; then
    RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
  fi
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
PRED_DIR="${RUN_DIR}/predictions"
MANIFEST_PATH="${RUN_DIR}/predictions_manifest.csv"

if [[ -e "${RUN_DIR}" && "${ALLOW_EXISTING_RUN_DIR}" -ne 1 ]]; then
  die "Run directory already exists: ${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}/logs"

source "${VENV_ACTIVATE}"
BASE_PYTHONPATH="${PYTHONPATH:-}"
NORMALIZED_EXTRA_ARGS_JSON="$(normalize_extra_args_json "${EXTRA_ARGS_JSON}")"
EXTRA_ARGS_JSON="${NORMALIZED_EXTRA_ARGS_JSON}"
if [[ -n "${BASE_PYTHONPATH}" ]]; then
  export PYTHONPATH="${CODE_ROOT}:${BASE_PYTHONPATH}"
else
  export PYTHONPATH="${CODE_ROOT}"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction

if [[ -n "${BASE_PYTHONPATH}" ]]; then
  CHECKPOINT_PROFILE_PYTHONPATH="${CHECKPOINT_PROFILE_ROOT}:${BASE_PYTHONPATH}"
else
  CHECKPOINT_PROFILE_PYTHONPATH="${CHECKPOINT_PROFILE_ROOT}"
fi
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
  echo "input_root=${INPUT_ROOT}"
  echo "input_root_mode=${INPUT_ROOT_MODE}"
  echo "date_preset=${DATE_PRESET}"
  echo "dates=${DATES}"
  echo "steps=${STEPS}"
  echo "bundle_pairs=${BUNDLE_PAIRS:-none}"
  echo "members=${MEMBERS}"
  echo "output_predictions_dir=${PRED_DIR}"
  echo "allow_existing_predictions_dir=${ALLOW_EXISTING_PREDICTIONS_DIR}"
  echo "allow_overwrite_existing_prediction_files=${ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES}"
  echo "manifest=${MANIFEST_PATH}"
  echo "device=${DEVICE}"
  echo "precision=${PRECISION}"
  echo "validation_frequency=${VALIDATION_FREQUENCY}"
  echo "num_gpus_per_model=${NUM_GPUS_PER_MODEL}"
  echo "output_weather_state_mode=${OUTPUT_WEATHER_STATE_MODE}"
  echo "output_weather_states=${OUTPUT_WEATHER_STATES}"
  echo "slim_output=${SLIM_OUTPUT}"
  echo "extra_args_json=${EXTRA_ARGS_JSON}"
} > "${RUN_DIR}/manual_inference_run_info.txt"

python - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${RUN_ID}" "${RUN_DIR}" "${STACK_FLAVOR}" "${LANE}" "${SOURCE_HPC}" "${HOST_SHORT}" "${VENV_ACTIVATE}" "${CODE_ROOT}" "${CHECKPOINT_REF}" "${CKPT_ROOT}" "${INPUT_ROOT}" "${INPUT_ROOT_MODE}" "${DATES}" "${MEMBERS}" "${STEPS}" "${BUNDLE_PAIRS}" "${OUTPUT_WEATHER_STATE_MODE}" "${OUTPUT_WEATHER_STATES}" "${SLIM_OUTPUT}" "${EXTRA_ARGS_JSON}" <<'PY'
import json
import sys
from pathlib import Path


def parse_int_list(raw):
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_bundle_pairs(raw):
    pairs = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        date_raw, step_raw = token.split(":", 1)
        pairs.append(
            {
                "date": int(date_raw.strip()),
                "step_hours": int(step_raw.strip()),
            }
        )
    return pairs


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
input_root = sys.argv[12]
input_root_mode = sys.argv[13]
dates = parse_int_list(sys.argv[14])
members = parse_int_list(sys.argv[15])
steps = parse_int_list(sys.argv[16])
bundle_pairs = parse_bundle_pairs(sys.argv[17])
output_weather_state_mode = sys.argv[18]
output_weather_states = sys.argv[19]
slim_output = int(sys.argv[20])
sampling_config_json = sys.argv[21]

checkpoint_path = Path(checkpoint_ref)
if not checkpoint_path.is_absolute():
    checkpoint_path = Path(ckpt_root) / checkpoint_ref
checkpoint_token = checkpoint_path.parent.name or checkpoint_path.stem

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
        "input_root": input_root,
        "input_root_mode": input_root_mode,
        "rebuilt_truth_bundle_root": input_root_mode == "rebuilt_truth_bundle_root",
        "bundle_scope": {
            "dates": dates,
            "members": members,
            "steps_hours": steps,
            "selection_mode": "explicit_pairs" if bundle_pairs else "cartesian_dates_steps",
        },
        "output_weather_state_mode": output_weather_state_mode,
        "output_weather_states": output_weather_states,
        "slim_output": bool(slim_output),
    },
}
if bundle_pairs:
    config["source"]["bundle_scope"]["bundle_pairs"] = bundle_pairs
try:
    config["model"] = {"development_hacks": {"extra_args": json.loads(sampling_config_json)}}
except json.JSONDecodeError:
    pass

out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

# Enrich EXPERIMENT_CONFIG with training params from MLflow (best-effort)
if [[ -n "${MLFLOW_EXPERIMENT_DIR}" && -d "${MLFLOW_EXPERIMENT_DIR}" ]]; then
  python - "${RUN_DIR}/EXPERIMENT_CONFIG.yaml" "${CHECKPOINT_REF}" "${CKPT_ROOT}" "${MLFLOW_EXPERIMENT_DIR}" <<'PYENRICH'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
checkpoint_ref = sys.argv[2]
ckpt_root = sys.argv[3]
mlflow_dir = Path(sys.argv[4])

checkpoint_path = Path(checkpoint_ref)
if not checkpoint_path.is_absolute():
    checkpoint_path = Path(ckpt_root) / checkpoint_ref
mlflow_run_id = checkpoint_path.parent.name

run_dir = mlflow_dir / mlflow_run_id
if not run_dir.is_dir():
    print(f"[enrich] MLflow run dir not found: {run_dir}, skipping enrichment")
    raise SystemExit(0)

def read_param(name, default=None):
    p = run_dir / "params" / name
    if p.is_file():
        return p.read_text().strip()
    return default

def read_param_float(name, default=None):
    v = read_param(name)
    if v is not None:
        try:
            return float(v)
        except ValueError:
            pass
    return default

def read_param_int(name, default=None):
    v = read_param(name)
    if v is not None:
        try:
            return int(v)
        except ValueError:
            pass
    return default

config = json.loads(config_path.read_text())

config["training"] = {
    "max_steps": read_param_int("config.training.max_steps"),
    "lr": {
        "rate": read_param_float("config.training.lr.rate"),
        "iterations": read_param_int("config.training.lr.iterations"),
        "warmup": read_param_int("config.training.lr.warmup"),
        "min": read_param_float("config.training.lr.min"),
    },
    "training_approach": read_param("config.training.training_approach"),
    "transfer_learning": read_param("config.training.transfer_learning"),
}

config["dataloader"] = {
    "batch_size_training": read_param_int("config.dataloader.batch_size.training"),
    "dataloader_config": read_param("config.dataloader.config_name",
                                     read_param("config.dataloader")),
}
select_forcings = read_param("config.dataloader.select_forcings")
if select_forcings:
    config["dataloader"]["select_forcings"] = select_forcings

config["diffusion_defaults"] = {
    "log_normal_mean": read_param_float("config.model.model.diffusion.log_normal_mean"),
    "log_normal_std": read_param_float("config.model.model.diffusion.log_normal_std"),
    "sigma_max": read_param_float("config.model.model.diffusion.sigma_max"),
    "sigma_min": read_param_float("config.model.model.diffusion.sigma_min"),
}

config["hardware"] = {
    "num_gpus_per_model": read_param_int("config.hardware.num_gpus_per_model"),
    "num_gpus_per_node": read_param_int("config.hardware.num_gpus_per_node"),
    "num_nodes": read_param_int("config.hardware.num_nodes"),
}

meta_yaml = run_dir / "meta.yaml"
if meta_yaml.is_file():
    run_name = None
    experiment_id = None
    for line in meta_yaml.read_text().splitlines():
        if line.startswith("run_name:"):
            run_name = line.split(":", 1)[1].strip().strip("'\"")
        elif line.startswith("experiment_id:"):
            experiment_id = line.split(":", 1)[1].strip().strip("'\"")
    config["mlflow"] = {
        "run_id": mlflow_run_id,
        "run_name": run_name,
        "experiment_id": experiment_id,
    }

# Remove None values for cleaner output
def remove_nones(obj):
    if isinstance(obj, dict):
        return {k: remove_nones(v) for k, v in obj.items() if v is not None}
    return obj

config = remove_nones(config)
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(f"[enrich] EXPERIMENT_CONFIG enriched with training params from MLflow run {mlflow_run_id}")
PYENRICH
fi

cd "${CODE_ROOT}"

PREDICTION_OVERWRITE_FLAGS=()
if [[ "${ALLOW_EXISTING_PREDICTIONS_DIR}" -eq 1 ]]; then
  PREDICTION_OVERWRITE_FLAGS+=(--allow-existing-out-dir)
fi
if [[ "${ALLOW_OVERWRITE_EXISTING_PREDICTION_FILES}" -eq 1 ]]; then
  PREDICTION_OVERWRITE_FLAGS+=(--allow-overwrite-existing-files)
fi

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  NEW_STACK_OUTPUT_FLAGS=()
  if [[ -n "${OUTPUT_WEATHER_STATES}" ]]; then
    NEW_STACK_OUTPUT_FLAGS+=(--output-weather-states "${OUTPUT_WEATHER_STATES}")
  fi
  if [[ "${SLIM_OUTPUT}" -eq 1 ]]; then
    NEW_STACK_OUTPUT_FLAGS+=(--slim-output)
  fi
  if [[ "${NUM_GPUS_PER_MODEL}" -gt 1 ]]; then
    srun --ntasks="${NUM_GPUS_PER_MODEL}" --ntasks-per-node="${NUM_GPUS_PER_MODEL}" \
      python -m eval.predict.main \
      --input-root "${INPUT_ROOT}" \
      --input-root-mode "${INPUT_ROOT_MODE}" \
      --out-dir "${PRED_DIR}" \
      --name-ckpt "${CHECKPOINT_REF}" \
      --ckpt-root "${CKPT_ROOT}" \
      "${PREDICTION_OVERWRITE_FLAGS[@]}" \
      --device "${DEVICE}" \
      --num-gpus-per-model "${NUM_GPUS_PER_MODEL}" \
      --precision "${PRECISION}" \
      --validation-frequency "${VALIDATION_FREQUENCY}" \
      --output-weather-state-mode "${OUTPUT_WEATHER_STATE_MODE}" \
      --members "${MEMBERS}" \
      --steps "${STEPS}" \
      --dates "${DATES}" \
      --bundle-pairs "${BUNDLE_PAIRS}" \
      --extra-args-json "${EXTRA_ARGS_JSON}" \
      "${NEW_STACK_OUTPUT_FLAGS[@]}" \
      --manifest "${MANIFEST_PATH}"
  else
    python -m eval.predict.main \
      --input-root "${INPUT_ROOT}" \
      --input-root-mode "${INPUT_ROOT_MODE}" \
      --out-dir "${PRED_DIR}" \
      --name-ckpt "${CHECKPOINT_REF}" \
      --ckpt-root "${CKPT_ROOT}" \
      "${PREDICTION_OVERWRITE_FLAGS[@]}" \
      --device "${DEVICE}" \
      --num-gpus-per-model "${NUM_GPUS_PER_MODEL}" \
      --precision "${PRECISION}" \
      --validation-frequency "${VALIDATION_FREQUENCY}" \
      --output-weather-state-mode "${OUTPUT_WEATHER_STATE_MODE}" \
      --members "${MEMBERS}" \
      --steps "${STEPS}" \
      --dates "${DATES}" \
      --bundle-pairs "${BUNDLE_PAIRS}" \
      --extra-args-json "${EXTRA_ARGS_JSON}" \
      "${NEW_STACK_OUTPUT_FLAGS[@]}" \
      --manifest "${MANIFEST_PATH}"
  fi
else
  OLD_ALLOW_FLAG=()
  if [[ "${ALLOW_MISSING_TARGET_OLD}" -eq 1 ]]; then
    OLD_ALLOW_FLAG=(--allow-missing-target)
  fi
  python -m eval.predict.main \
    --input-root "${INPUT_ROOT}" \
    --out-dir "${PRED_DIR}" \
    --ckpt-id "${CHECKPOINT_REF}" \
    --ckpt-root "${CKPT_ROOT}" \
    "${PREDICTION_OVERWRITE_FLAGS[@]}" \
    --device "${DEVICE}" \
    --precision "${PRECISION}" \
    --members "${MEMBERS}" \
    --steps "${STEPS}" \
    --dates "${DATES}" \
    --bundle-pairs "${BUNDLE_PAIRS}" \
    --extra-args-json "${EXTRA_ARGS_JSON}" \
    --manifest "${MANIFEST_PATH}" \
    "${OLD_ALLOW_FLAG[@]}"
fi

python - "${PRED_DIR}" "${STACK_FLAVOR}" "${ALLOW_MISSING_TARGET_OLD}" <<'PY'
import pathlib
import sys
import xarray as xr

pred_dir = pathlib.Path(sys.argv[1])
stack = sys.argv[2]
allow_missing_target_old = int(sys.argv[3])
paths = sorted(pred_dir.glob("predictions_*_step*.nc"))
if not paths:
    raise SystemExit(f"No predictions_*.nc files found in {pred_dir}")

missing_y_pred = []
missing_y = []
for p in paths:
    with xr.open_dataset(p) as ds:
        if "y_pred" not in ds.variables:
            missing_y_pred.append(str(p))
        if stack == "new":
            if "y" not in ds.variables:
                missing_y.append(str(p))
        elif allow_missing_target_old == 0:
            if "y" not in ds.variables:
                missing_y.append(str(p))

if missing_y_pred:
    raise SystemExit(f"Missing y_pred in files: {missing_y_pred[:5]}")
if missing_y:
    raise SystemExit(f"Missing y in files: {missing_y[:5]}")

print(f"Validated {len(paths)} predictions files in {pred_dir}")
PY

echo "DONE: ${PRED_DIR}"
