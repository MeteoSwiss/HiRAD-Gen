#!/bin/bash --login
# DEPRECATED: This template is not called by any eval flow.
# All lanes now use shared templates + finalize_lean_eval_layout for canonical output.
# Kept for reference only. Do not modify.
#
# (Original description) Post-prediction writer for o48->o96 evaluation runs.
# Reuses the shared spectra, TC, and surface-loss helpers but writes a compact
# o48_o96 metrics sidecar for the Humberto and validation-tail workflow.
#SBATCH --job-name=o48_o96_write
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ID="REPLACE_RUN_ID"
OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_ROOT=""
PRED_DIR=""
EXPECTED_PRED_FILES=10
SIGMA_RUN_ID=""

TC_EVENTS="humberto"
TC_SUPPORT_MODE="regridded"      # native | regridded
TC_DISPLAY_LABEL=""              # optional shorter legend label; blank => RUN_ID
BASE_TC_DIR="/home/ecm5702/perm/reference/o48_o96/tc"

SPECTRA_WEATHER_STATES="10u,10v,2t,msl,t_850,z_500"
SPECTRA_NSIDE=64
SPECTRA_LMAX=95
SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE=0
SPECTRA_MEMBER_AGG="per-file-mean"   # per-file-mean | raw-members
PREDICTION_VAR="y_pred"              # y_pred | x_interp | any high-res prediction-like var
TRUTH_VAR="y"
SPECTRA_OUT_DIRNAME="spectra_o48_o96"
SURFACE_JSON_NAME="surface_loss_summary.json"
METRICS_JSON_NAME="o48_o96_metrics.json"
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

[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID in USER SETTINGS."
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"
require_choice "${TC_SUPPORT_MODE}" native regridded
require_choice "${SPECTRA_MEMBER_AGG}" per-file-mean raw-members
[[ -n "${PREDICTION_VAR// /}" ]] || die "PREDICTION_VAR must be non-empty"
[[ -n "${TRUTH_VAR// /}" ]] || die "TRUTH_VAR must be non-empty"

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
fi
if [[ -z "${PRED_DIR}" ]]; then
  PRED_DIR="${RUN_ROOT}/predictions"
fi

[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT not found: ${RUN_ROOT}"
[[ -d "${PRED_DIR}" ]] || die "PRED_DIR not found: ${PRED_DIR}"
[[ "${EXPECTED_PRED_FILES}" =~ ^[0-9]+$ ]] || die "EXPECTED_PRED_FILES must be an integer"

pred_count=$(find "${PRED_DIR}" -maxdepth 1 -name 'predictions_*.nc' | wc -l)
[[ "${pred_count}" -eq "${EXPECTED_PRED_FILES}" ]] || die \
  "Expected ${EXPECTED_PRED_FILES} prediction files, found ${pred_count} under ${PRED_DIR}"

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) VENV_ACTIVATE="/home/ecm5702/dev/.ds-dyn/bin/activate" ;;
  ag*) VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag/bin/activate" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

REPO_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
TEMPLATE_DIR="${REPO_ROOT}/eval/jobs/templates"
SPECTRA_SCRIPT="${TEMPLATE_DIR}/predictions_dir_spectra.py"
SURFACE_SCRIPT="${REPO_ROOT}/eval/jobs/scoreboard_surface_loss.py"

[[ -f "${SPECTRA_SCRIPT}" ]] || die "Missing spectra script: ${SPECTRA_SCRIPT}"
[[ -f "${SURFACE_SCRIPT}" ]] || die "Missing surface-loss script: ${SURFACE_SCRIPT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"

source "${VENV_ACTIVATE}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

TC_EVENT_TAG=""
TC_STATS=""
if [[ -n "${TC_EVENTS// /}" ]]; then
  if [[ "${TC_SUPPORT_MODE}" == "regridded" ]]; then
    module load ecmwf-toolbox
    METVIEW_BIN="$(command -v metview || true)"
    if [[ -n "${METVIEW_BIN}" ]]; then
      export PATH="$(dirname "${METVIEW_BIN}"):${PATH}"
    fi
    python -c "import metview" >/dev/null 2>&1 || \
      die "TC_SUPPORT_MODE=regridded requires metview. Use AC or switch to native mode."
  fi
  TC_EVENT_TAG="${TC_EVENTS//,/_}"
  TC_EVENT_TAG="${TC_EVENT_TAG// /}"
  TC_OUT_NAME="tc_normed_pdfs_${TC_EVENT_TAG}_${RUN_ID}_from_predictions.pdf"
  if [[ -n "${TC_DISPLAY_LABEL}" ]]; then
    python -m eval._backends.tc.workflows pdf \
      --predictions-dir "${PRED_DIR}" \
      --outdir "${RUN_ROOT}" \
      --run-label "${RUN_ID}" \
      --display-label "${TC_DISPLAY_LABEL}" \
      --out-name "${TC_OUT_NAME}" \
      --base-tc-dir "${BASE_TC_DIR}" \
      --support-mode "${TC_SUPPORT_MODE}" \
      --events "${TC_EVENTS}" \
      --prediction-var "${PREDICTION_VAR}"
  else
    python -m eval._backends.tc.workflows pdf \
      --predictions-dir "${PRED_DIR}" \
      --outdir "${RUN_ROOT}" \
      --run-label "${RUN_ID}" \
      --out-name "${TC_OUT_NAME}" \
      --base-tc-dir "${BASE_TC_DIR}" \
      --support-mode "${TC_SUPPORT_MODE}" \
      --events "${TC_EVENTS}" \
      --prediction-var "${PREDICTION_VAR}"
  fi
  TC_STATS="${RUN_ROOT}/${TC_OUT_NAME%.pdf}.stats.json"
fi

SPECTRA_OUT="${RUN_ROOT}/${SPECTRA_OUT_DIRNAME}"
SURFACE_JSON="${RUN_ROOT}/${SURFACE_JSON_NAME}"
METRICS_JSON="${RUN_ROOT}/${METRICS_JSON_NAME}"

mkdir -p "${SPECTRA_OUT}"

python "${SPECTRA_SCRIPT}" \
  --predictions-dir "${PRED_DIR}" \
  --out-dir "${SPECTRA_OUT}" \
  --run-label "${RUN_ID}" \
  --weather-states "${SPECTRA_WEATHER_STATES}" \
  --nside "${SPECTRA_NSIDE}" \
  --lmax "${SPECTRA_LMAX}" \
  --score-wavenumber-min-exclusive "${SPECTRA_SCORE_WAVENUMBER_MIN_EXCLUSIVE}" \
  --member-aggregation "${SPECTRA_MEMBER_AGG}" \
  --prediction-var "${PREDICTION_VAR}"

python "${SURFACE_SCRIPT}" \
  --predictions-dir "${PRED_DIR}" \
  --out-json "${SURFACE_JSON}" \
  --prediction-var "${PREDICTION_VAR}" \
  --truth-var "${TRUTH_VAR}"

python - "${RUN_ID}" "${OUTPUT_ROOT}" "${SIGMA_RUN_ID}" "${TC_STATS}" "${SPECTRA_OUT}" "${SURFACE_JSON}" "${METRICS_JSON}" "${PRED_DIR}" "${EXPECTED_PRED_FILES}" "${TC_EVENTS}" "${PREDICTION_VAR}" "${TRUTH_VAR}" <<'PY'
import json
import sys
from pathlib import Path

from eval.jobs import scoreboard_metrics as metrics

run_id = sys.argv[1]
output_root = Path(sys.argv[2])
sigma_run_id = sys.argv[3]
tc_stats_path = Path(sys.argv[4]) if sys.argv[4] else Path()
spectra_dir = Path(sys.argv[5])
surface_json_path = Path(sys.argv[6])
out_path = Path(sys.argv[7])
pred_dir = Path(sys.argv[8])
expected_pred_files = int(sys.argv[9])
requested_events = tuple(event.strip().lower() for event in sys.argv[10].split(",") if event.strip())
prediction_var = sys.argv[11]
truth_var = sys.argv[12]

summary = metrics.build_run_scoreboard_metrics(
    run_id=run_id,
    output_root=output_root,
    sigma_run_id=sigma_run_id,
    tc_stats_path=tc_stats_path,
    spectra_dir=spectra_dir,
    surface_json_path=surface_json_path,
    event_names=requested_events or None,
)
summary["lane"] = "o48_o96"
summary["predictions_dir"] = str(pred_dir)
summary["prediction_var"] = prediction_var
summary["truth_var"] = truth_var
summary["baseline_mode"] = "x_interp" if prediction_var == "x_interp" else "model_prediction"
if prediction_var == "x_interp":
    summary["spectra_full_field_mean_relative_l2"] = summary.get("spectra_mean_relative_l2")
    summary["spectra_mean_relative_l2"] = None
    summary["spectra_score_excluded_reason"] = (
        "x_interp full-field spectra are dominated by large-scale input power and are not a "
        "downscaling skill score; residual prediction spectrum is zero by construction."
    )
summary["expected_prediction_files"] = expected_pred_files
summary["requested_tc_events"] = list(requested_events)
summary["artifacts"] = {
    "tc_stats": str(tc_stats_path) if str(tc_stats_path) and tc_stats_path.exists() else None,
    "spectra_dir": str(spectra_dir),
    "surface_loss_summary": str(surface_json_path),
}

spectra_metrics = metrics.load_spectra_metrics(spectra_dir)
summary["spectra_coverage"] = spectra_metrics.get("coverage")
summary["spectra_n_curves"] = spectra_metrics.get("n_curves")

if len(requested_events) == 1:
    only_event = requested_events[0]
    only_score = summary.get("tc_extreme_scores", {}).get(only_event)
    if only_score is not None:
        summary[f"{only_event}_tc_extreme_score"] = only_score

out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "=== o48->o96 writer complete ==="
echo "run_id=${RUN_ID}"
echo "pred_dir=${PRED_DIR}"
echo "pred_count=${pred_count}"
echo "prediction_var=${PREDICTION_VAR}"
echo "truth_var=${TRUTH_VAR}"
echo "spectra_out=${SPECTRA_OUT}"
echo "surface_json=${SURFACE_JSON}"
echo "metrics_json=${METRICS_JSON}"
if [[ -n "${TC_STATS}" ]]; then
  echo "tc_stats=${TC_STATS}"
else
  echo "tc_stats=skipped"
fi
