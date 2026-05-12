#!/bin/bash --login
# Canonical MLflow training-loss plotting template.
# Generates exact-run loss curves, overview plots, and per-variable panels from local MLflow logs.
# If no matching run is found, it writes a status JSON instead of failing hard.
#SBATCH --job-name=mlflow_loss_plots
#SBATCH --qos=nf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
MLFLOW_EXPERIMENT_DIR="/home/ecm5702/scratch/aifs/logs/mlflow/909682684414341917"
MIN_STEPS=0
NAME_FILTER=""
STATUS_JSON="${RUN_ROOT}/training_loss_plots_status.json"
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*)
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag*)
    source /home/ecm5702/dev/.ds-ag/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
  *)
    die "Unknown cluster from hostname '${HOST_SHORT}'."
    ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${MLFLOW_EXPERIMENT_DIR}" ]] || die "MLFLOW_EXPERIMENT_DIR does not exist: ${MLFLOW_EXPERIMENT_DIR}"
[[ "${MIN_STEPS}" =~ ^[0-9]+$ ]] || die "MIN_STEPS must be a non-negative integer."

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 - "${MLFLOW_EXPERIMENT_DIR}" "${MIN_STEPS}" "${NAME_FILTER}" "${STATUS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/mlflow")
from loader import filter_runs, load

experiment_dir = Path(sys.argv[1])
min_steps = int(sys.argv[2])
name_filter_spec = sys.argv[3]
status_path = Path(sys.argv[4])

runs = load(experiment_dir)
candidate_filters = []
for token in name_filter_spec.split(","):
    token = token.strip()
    if token and token not in candidate_filters:
        candidate_filters.append(token)
if not candidate_filters:
    candidate_filters = [None]

selected_filter = None
for candidate in candidate_filters:
    matched_runs = filter_runs(runs, min_steps=min_steps, name_contains=candidate)
    if matched_runs:
        selected_filter = candidate
        runs = matched_runs
        break
else:
    runs = filter_runs(runs, min_steps=min_steps, name_contains=candidate_filters[0])

payload = {
    "experiment_dir": str(experiment_dir),
    "min_steps": min_steps,
    "name_filter": selected_filter,
    "name_filter_candidates": candidate_filters,
    "matched_runs": sorted(runs),
    "matched_count": len(runs),
}
status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

MATCHED_COUNT="$("${PYTHON}" - "${STATUS_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["matched_count"])
PY
)"

if [[ "${MATCHED_COUNT}" -eq 0 ]]; then
  echo "[INFO] No matching MLflow runs found; leaving status JSON only: ${STATUS_JSON}"
  exit 0
fi

SELECTED_NAME_FILTER="$("${PYTHON}" - "${STATUS_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["name_filter"] or "")
PY
)"

echo "[INFO] cluster=${HOST_SHORT}"
echo "[INFO] matched_runs=${MATCHED_COUNT}"
echo "[INFO] name_filter=${SELECTED_NAME_FILTER}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

"${PYTHON}" "${PROJECT_ROOT}/mlflow/plot.py" \
  "${MLFLOW_EXPERIMENT_DIR}" \
  "${MIN_STEPS}" \
  "${SELECTED_NAME_FILTER}" \
  "${RUN_ROOT}"

"${PYTHON}" "${PROJECT_ROOT}/mlflow/plot_loss.py" \
  "${MLFLOW_EXPERIMENT_DIR}" \
  "${SELECTED_NAME_FILTER}" \
  "${RUN_ROOT}/loss_curves.png"

[[ -f "${RUN_ROOT}/key_vars.png" ]] || die "Expected MLflow plot missing: ${RUN_ROOT}/key_vars.png"
[[ -f "${RUN_ROOT}/overview.png" ]] || die "Expected MLflow plot missing: ${RUN_ROOT}/overview.png"
[[ -f "${RUN_ROOT}/all_vars.png" ]] || die "Expected MLflow plot missing: ${RUN_ROOT}/all_vars.png"
[[ -f "${RUN_ROOT}/loss_curves.png" ]] || die "Expected MLflow plot missing: ${RUN_ROOT}/loss_curves.png"

echo "DONE: ${RUN_ROOT}/key_vars.png"
echo "DONE: ${RUN_ROOT}/overview.png"
echo "DONE: ${RUN_ROOT}/all_vars.png"
echo "DONE: ${RUN_ROOT}/loss_curves.png"
