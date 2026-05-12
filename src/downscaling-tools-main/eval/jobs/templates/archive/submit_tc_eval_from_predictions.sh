#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_PATH="${1:-${SCRIPT_DIR}/tc_eval_from_predictions.sbatch}"
REQUESTED_PROFILE="${TC_SUBMIT_PROFILE:-auto}"   # auto | raw | ac_cpu_safe | ag_cpu_safe
RENDER_ROOT="${TC_SUBMIT_RENDER_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
QOS_OVERRIDE="${TC_SUBMIT_QOS_OVERRIDE:-}"
MEM_OVERRIDE="${TC_SUBMIT_MEM_OVERRIDE:-}"
TIME_OVERRIDE="${TC_SUBMIT_TIME_OVERRIDE:-}"
GPU_OVERRIDE="${TC_SUBMIT_GPUS_OVERRIDE:-}"
NO_SUBMIT="${TC_SUBMIT_NO_SUBMIT:-0}"
HOLD="${TC_SUBMIT_HOLD:-0}"

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

require_bool() {
  local value="$1"
  [[ "${value}" == "0" || "${value}" == "1" ]] || die "Expected 0 or 1, got '${value}'"
}

[[ -f "${TEMPLATE_PATH}" ]] || die "Template not found: ${TEMPLATE_PATH}"
require_choice "${REQUESTED_PROFILE}" auto raw ac_cpu_safe ag_cpu_safe
require_bool "${NO_SUBMIT}"
require_bool "${HOLD}"

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

case "${REQUESTED_PROFILE}" in
  auto)
    if [[ "${HOST_FAMILY}" == "ac" ]]; then
      RESOLVED_PROFILE="ac_cpu_safe"
    else
      RESOLVED_PROFILE="ag_cpu_safe"
    fi
    ;;
  *)
    RESOLVED_PROFILE="${REQUESTED_PROFILE}"
    ;;
esac

EFFECTIVE_QOS=""
GPU_MODE="keep"
GPU_VALUE=""
case "${RESOLVED_PROFILE}" in
  raw)
    ;;
  ac_cpu_safe)
    EFFECTIVE_QOS="nf"
    GPU_MODE="drop"
    ;;
  ag_cpu_safe)
    EFFECTIVE_QOS="ng"
    GPU_MODE="set"
    GPU_VALUE="0"
    ;;
esac

if [[ -n "${QOS_OVERRIDE}" ]]; then
  EFFECTIVE_QOS="${QOS_OVERRIDE}"
fi
if [[ -n "${GPU_OVERRIDE}" ]]; then
  GPU_MODE="set"
  GPU_VALUE="${GPU_OVERRIDE}"
fi

PREFLIGHT_SCRIPT="${SCRIPT_DIR}/preflight_eval_check.sh"
source "${PREFLIGHT_SCRIPT}"
preflight_cluster
if [[ "${PREFLIGHT_CLUSTER}" != "${HOST_FAMILY}" ]]; then
  die "Preflight cluster mismatch: expected ${HOST_FAMILY}, got ${PREFLIGHT_CLUSTER}"
fi
if [[ -n "${EFFECTIVE_QOS}" ]]; then
  preflight_qos "${EFFECTIVE_QOS}"
fi
preflight_walltime_guidance tc_eval
preflight_summary

RENDER_DIR="${RENDER_ROOT}/$(date -u +%Y%m%d)"
mkdir -p "${RENDER_DIR}"
STAMP="$(date -u +%H%M%SZ)"
BASE_NAME="$(basename "${TEMPLATE_PATH}" .sbatch)"
RENDERED_PATH="${RENDER_DIR}/${BASE_NAME}.${RESOLVED_PROFILE}.${STAMP}.sbatch"

python - "${TEMPLATE_PATH}" "${RENDERED_PATH}" "${EFFECTIVE_QOS}" "${GPU_MODE}" "${GPU_VALUE}" "${TIME_OVERRIDE}" "${MEM_OVERRIDE}" "${RESOLVED_PROFILE}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
effective_qos = sys.argv[3]
gpu_mode = sys.argv[4]
gpu_value = sys.argv[5]
time_override = sys.argv[6]
mem_override = sys.argv[7]
profile = sys.argv[8]

lines = src.read_text().splitlines(keepends=True)
rendered = []
timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

for idx, line in enumerate(lines):
    if idx == 0 and line.startswith("#!"):
        rendered.append(line)
        rendered.append(
            f"# Rendered from {src} via submit_tc_eval_from_predictions.sh profile={profile} at {timestamp}\n"
        )
        continue

    if line.startswith("#SBATCH --qos="):
        rendered.append(f"#SBATCH --qos={effective_qos}\n" if effective_qos else line)
        continue

    if line.startswith("#SBATCH --gpus-per-node="):
        if gpu_mode == "drop":
            continue
        if gpu_mode == "set":
            rendered.append(f"#SBATCH --gpus-per-node={gpu_value}\n")
            continue

    if line.startswith("#SBATCH --time=") and time_override:
        rendered.append(f"#SBATCH --time={time_override}\n")
        continue

    if line.startswith("#SBATCH --mem=") and mem_override:
        rendered.append(f"#SBATCH --mem={mem_override}\n")
        continue

    rendered.append(line)

dst.write_text("".join(rendered))
PY

echo "[tc-submit] template=${TEMPLATE_PATH}"
echo "[tc-submit] host=${HOST_SHORT}"
echo "[tc-submit] profile=${RESOLVED_PROFILE}"
echo "[tc-submit] rendered=${RENDERED_PATH}"
echo "[tc-submit] qos=${EFFECTIVE_QOS:-template-default}"
case "${GPU_MODE}" in
  keep) echo "[tc-submit] gpu_request=template-default" ;;
  drop) echo "[tc-submit] gpu_request=dropped" ;;
  set)  echo "[tc-submit] gpu_request=${GPU_VALUE}" ;;
esac
echo "[tc-submit] mem_override=${MEM_OVERRIDE:-none}"
echo "[tc-submit] time_override=${TIME_OVERRIDE:-none}"

if [[ "${NO_SUBMIT}" == "1" ]]; then
  echo "[tc-submit] render-only mode enabled; not submitting"
  exit 0
fi

SBATCH_ARGS=()
if [[ "${HOLD}" == "1" ]]; then
  SBATCH_ARGS+=(--hold)
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${RENDERED_PATH}")"
echo "[tc-submit] submitted job ${JOB_ID}"
