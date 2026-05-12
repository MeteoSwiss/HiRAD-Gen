#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
# Canonical proxy spectra template — works on both AC and AG.
#
# Computes the lightweight healpy-based spectra summary from an existing
# `predictions_*.nc` tree using:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/predictions_dir_spectra.py
#
# Preferred launch path for the `o320 -> o1280` manual-eval lane:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
#
# For the trusted AC-only ECMWF transform route, use:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/spectra_ecmwf_from_predictions.sbatch
#
# Raw submission compatibility note:
#   the directives below stay cross-cluster-compatible for deliberate direct
#   `sbatch`, but the preferred helper path patches them to a host-safe CPU
#   posture (`nf` without GPUs on AC, `ng` with `--gpus-per-node=0` on AG).

#SBATCH --job-name=spectra_proxy
#SBATCH --qos=ng
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_SOURCE_DIR="${RUN_ROOT}/predictions"
SUBSET_TAG="all_dates_all_steps"
SUBSET_DIR="${RUN_ROOT}/predictions_subset_${SUBSET_TAG}"
# Canonical output path (eval-data-contract)
OUT_DIR="${RUN_ROOT}/data/spectra"
DATE_LIST="ALL"                                          # CSV or ALL
STEP_LIST="ALL"                                          # CSV or ALL
WEATHER_STATES="10u,10v,2t,msl,t_850,z_500"
NSIDE=512
LMAX=1279
MEMBER_AGGREGATION="raw-members"                         # per-file-mean | raw-members
SHOW_INDIVIDUAL_CURVES=0                                 # 1 => overlay every case curve
WRITE_EXPERIMENT_CONFIG=0                                # 1 => write ${OUT_DIR}/EXPERIMENT_CONFIG.yaml
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
SPECTRA_PY="${PROJECT_ROOT}/eval/jobs/templates/predictions_dir_spectra.py"

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

csv_matches() {
  local csv="$1"
  local needle="$2"
  if [[ "${csv}" == "ALL" ]]; then
    return 0
  fi
  [[ ",${csv}," == *",${needle},"* ]]
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *) die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

require_choice "${MEMBER_AGGREGATION}" per-file-mean raw-members
[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${PREDICTIONS_SOURCE_DIR}" ]] || die "Predictions source directory does not exist: ${PREDICTIONS_SOURCE_DIR}"
[[ -f "${SPECTRA_PY}" ]] || die "Missing helper script: ${SPECTRA_PY}"

mkdir -p "${RUN_ROOT}/logs" "${SUBSET_DIR}" "${OUT_DIR}"
rm -f "${SUBSET_DIR}"/predictions_*.nc

count=0
for file in "${PREDICTIONS_SOURCE_DIR}"/predictions_*.nc; do
  [[ -f "${file}" ]] || continue
  base="$(basename "${file}")"
  if [[ "${base}" =~ ^predictions_([0-9]{8})_step([0-9]{2,3})\.nc$ ]]; then
    date_key="${BASH_REMATCH[1]}"
    step_key="$((10#${BASH_REMATCH[2]}))"
    if csv_matches "${DATE_LIST}" "${date_key}" && csv_matches "${STEP_LIST}" "${step_key}"; then
      ln -sf "${file}" "${SUBSET_DIR}/${base}"
      count=$((count + 1))
    fi
  fi
done

[[ "${count}" -gt 0 ]] || die "No predictions matched DATE_LIST=${DATE_LIST} STEP_LIST=${STEP_LIST}"

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

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_LABEL="${RUN_ID}_${SUBSET_TAG}"

echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] matched_prediction_files=${count}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

CMD=(
  "${PYTHON}" "${SPECTRA_PY}"
  --predictions-dir "${SUBSET_DIR}"
  --out-dir "${OUT_DIR}"
  --run-label "${RUN_LABEL}"
  --weather-states "${WEATHER_STATES}"
  --nside "${NSIDE}"
  --lmax "${LMAX}"
  --member-aggregation "${MEMBER_AGGREGATION}"
)

if [[ "${SHOW_INDIVIDUAL_CURVES}" -eq 1 ]]; then
  CMD+=(--show-individual-curves)
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

SUMMARY_JSON="${OUT_DIR}/spectra_summary.json"
[[ -f "${SUMMARY_JSON}" ]] || die "Expected spectra summary missing: ${SUMMARY_JSON}"

# Canonical alias: summary.json (eval-data-contract expects data/spectra/summary.json)
if [[ -f "${SUMMARY_JSON}" ]] && [[ ! -e "${OUT_DIR}/summary.json" ]]; then
  ln -sfn "spectra_summary.json" "${OUT_DIR}/summary.json"
fi

# Legacy compat symlink at old location
ln -sfn "data/spectra" "${RUN_ROOT}/spectra_proxy_${SUBSET_TAG}" 2>/dev/null || true

# --- PDF generation: always attempted after successful compute (non-fatal) ---
PLOT_PDF_PY="${PROJECT_ROOT}/eval/jobs/templates/spectra_plot_pdf.py"
CURVE_SUMMARY="${OUT_DIR}/spectra_curve_summary.json"
# Also check canonical curves.json name
if [[ ! -f "${CURVE_SUMMARY}" ]] && [[ -f "${OUT_DIR}/curves.json" ]]; then
  CURVE_SUMMARY="${OUT_DIR}/curves.json"
fi
CONSOLIDATED_PDF="${RUN_ROOT}/spectra_proxy.pdf"
if [[ -f "${PLOT_PDF_PY}" ]] && [[ -f "${CURVE_SUMMARY}" ]]; then
  echo "[INFO] Generating consolidated PDF from curve summary..."
  "${PYTHON}" "${PLOT_PDF_PY}" \
    --spectra-dir "${OUT_DIR}" \
    --out-pdf "${CONSOLIDATED_PDF}" \
    || echo "[WARN] PDF generation failed. Re-run: ${PYTHON} ${PLOT_PDF_PY} --spectra-dir ${OUT_DIR} --out-pdf ${CONSOLIDATED_PDF}"
else
  echo "[WARN] Skipping PDF generation: spectra_plot_pdf.py or spectra_curve_summary.json not found."
fi

if [[ "${WRITE_EXPERIMENT_CONFIG}" -eq 1 ]]; then
  cat > "${OUT_DIR}/EXPERIMENT_CONFIG.yaml" <<EOF
run_id: ${RUN_ID}
purpose: "Prediction-run proxy spectra bundle"
predictions:
  source_dir: ${PREDICTIONS_SOURCE_DIR}
  filtered_subset_dir: ${SUBSET_DIR}
  date_list: ${DATE_LIST}
  step_list: ${STEP_LIST}
spectra:
  script: ${SPECTRA_PY}
  output_dir: ${OUT_DIR}
  summary_json: ${SUMMARY_JSON}
  weather_states: ${WEATHER_STATES}
  nside: ${NSIDE}
  lmax: ${LMAX}
  member_aggregation: ${MEMBER_AGGREGATION}
EOF
fi

echo "DONE: ${SUMMARY_JSON}"
