#!/bin/bash --login
# Canonical ECMWF spectra template — AC-only (requires gptosp.ser).
#
# Computes trusted spectral energy spectra from prediction NetCDF files using
# the ECMWF spectral-transform route (gptosp), following the same pipeline as:
#   <repo>/eval/spectra/grb_to_spectra.sh
#
# Pipeline:
#   1. Stage prediction fields as nopoles GRIB files (from predictions_*.nc)
#   2. Convert GRIBs to spectral harmonics via gptosp.ser
#   3. Compute spectra amplitudes from the staged spectral harmonics
#
# For the AG-side healpy proxy alternative, use:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/spectra_proxy_from_predictions.sbatch
#
# IMPORTANT: This template is AC-only. gptosp.ser and the required modules
# (ifs, eclib, pifsenv) are not available on AG nodes.
#
# Resource baselines (empirical):
#   300 transforms (6 params x 50 files): ~24h
#   Full pipeline with compute: 48:00:00 recommended
#   Memory: 128G sufficient
#
# Resumability: The template skips already-completed gptosp transforms
# (checks for existing *_sh files) so it can be resubmitted after walltime.

#SBATCH --job-name=spectra_ecmwf
#SBATCH --qos=nf
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"

# Date/step/member filters (CSV or ALL)
DATE_LIST="20230826,20230827,20230828,20230829,20230830"
STEP_LIST="120"
MEMBER_LIST="ALL"                # CSV of member indices (1-based) or ALL

# Weather states to compute spectra for
WEATHER_STATES="10u,10v,2t,sp,t_850,z_500"
TEMPLATE_ROOT="/home/ecm5702/perm/reference/o320_o1280/spectra/enfo_o1280"
TEMPLATE_GRIB_ROOT=""

# Output layout — canonical (eval-data-contract)
SPECTRA_ROOT="${RUN_ROOT}/data/spectra"
GRB_DIR="${SPECTRA_ROOT}/grb"
SH_DIR="${SPECTRA_ROOT}/spectral_harmonics"
SPECTRA_DIR="${SPECTRA_ROOT}/spectra"

# Conda env for step 3 (compute_spectra-3.py)
CONDA_ENV="/home/ecm5702/dev/.ds-dyn/bin/activate"

# Compute spectra script
COMPUTE_SPECTRA_PY="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/compute_prediction_spectra_amplitudes.py"

WRITE_EXPERIMENT_CONFIG=0
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

csv_matches() {
  local csv="$1"
  local needle="$2"
  if [[ "${csv}" == "ALL" ]]; then return 0; fi
  [[ ",${csv}," == *",${needle},"* ]]
}

# --- Cluster check ---
HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) ;;
  *) die "This template is AC-only (requires gptosp.ser). Current host: ${HOST_SHORT}" ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"

mkdir -p "${GRB_DIR}" "${SH_DIR}" "${SPECTRA_DIR}" "${RUN_ROOT}/logs"

# Legacy compat symlink
ln -sfn "data/spectra" "${RUN_ROOT}/spectra_ecmwf" 2>/dev/null || true

echo "=== STEP 1: Stage prediction fields as nopoles GRIBs ==="

# Use a short TMPDIR symlink to avoid gptosp path-length issues
WORK_LINK="${TMPDIR:-/tmp}/spectra_${SLURM_JOB_ID:-$$}"
ln -sfn "${SPECTRA_ROOT}" "${WORK_LINK}"
SHORT_GRB="${WORK_LINK}/grb"
SHORT_SH="${WORK_LINK}/spectral_harmonics"

source "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON="$(which python)"
echo "[INFO] Python: ${PYTHON} ($(${PYTHON} -V 2>&1))"

STAGE_ARGS=(
  --predictions-dir "${PREDICTIONS_DIR}"
  --out-dir "${GRB_DIR}"
  --template-root "${TEMPLATE_ROOT}"
  --weather-states "${WEATHER_STATES}"
  --date-list "${DATE_LIST}"
  --step-list "${STEP_LIST}"
  --member-list "${MEMBER_LIST}"
  --summary-path "${SPECTRA_ROOT}/staging_summary.json"
)
if [[ -n "${TEMPLATE_GRIB_ROOT}" ]]; then
  STAGE_ARGS+=(--template-grib-root "${TEMPLATE_GRIB_ROOT}")
fi
"${PYTHON}" "${PROJECT_ROOT}/eval/jobs/templates/stage_prediction_spectra_gribs.py" "${STAGE_ARGS[@]}"

echo ""
echo "=== STEP 2: Convert GRIBs to spectral harmonics via gptosp.ser ==="

module unload ecmwf-toolbox 2>/dev/null || true
module load eclib 2>/dev/null || true
module load pifsenv 2>/dev/null || true
module load ifs 2>/dev/null || true
export DR_HOOK_ASSERT_MPI_INITIALIZED=0

TRANSFORMS_DONE=0
TRANSFORMS_SKIPPED=0

mapfile -t PARAM_DIRS < <("${PYTHON}" - "${WEATHER_STATES}" <<'PY'
import sys
mapping = {
    "10u": "10u_sfc",
    "10v": "10v_sfc",
    "2t": "2t_sfc",
    "sp": "sp_sfc",
    "msl": "msl_sfc",
    "t_850": "t_850",
    "z_500": "z_500",
}
for token in sys.argv[1].split(","):
    token = token.strip()
    if token:
        print(mapping[token])
PY
)
for param_dir in "${PARAM_DIRS[@]}"; do
  mkdir -p "${SH_DIR}/${param_dir}"

  for grb_file in "${GRB_DIR}/${param_dir}"/*.grb; do
    [[ -f "${grb_file}" ]] || continue
    grb_base="$(basename "${grb_file}")"
    sh_out="${SH_DIR}/${param_dir}/${grb_base}_sh"

    if [[ -f "${sh_out}" ]] && [[ -s "${sh_out}" ]]; then
      TRANSFORMS_SKIPPED=$((TRANSFORMS_SKIPPED + 1))
      continue
    fi

    # Use short paths to avoid gptosp filename truncation
    short_in="${SHORT_GRB}/${param_dir}/${grb_base}"
    short_out="${SHORT_SH}/${param_dir}/${grb_base}_sh"

    echo "[gptosp] ${grb_base} -> ${grb_base}_sh"
    gptosp.ser -l -g "${short_in}" -S "${short_out}" && \
      TRANSFORMS_DONE=$((TRANSFORMS_DONE + 1)) || \
      echo "[WARN] gptosp failed for ${grb_base}"
  done
done

echo "[INFO] gptosp transforms done: ${TRANSFORMS_DONE}, skipped (existing): ${TRANSFORMS_SKIPPED}"

echo ""
echo "=== STEP 3: Compute spectra amplitudes ==="

module unload ifs 2>/dev/null || true
module load ecmwf-toolbox 2>/dev/null || true
METVIEW_BIN="$(command -v metview || true)"
[[ -n "${METVIEW_BIN}" ]] || die "metview not found after loading ecmwf-toolbox"

source "${CONDA_ENV}"
export PATH="$(dirname "${METVIEW_BIN}"):${PATH}"
PYTHON="$(which python)"

"${PYTHON}" "${COMPUTE_SPECTRA_PY}" \
  --spectral-harmonics-dir "${SH_DIR}" \
  --out-dir "${SPECTRA_DIR}" \
  --weather-states "${WEATHER_STATES}" \
  --summary-path "${SPECTRA_ROOT}/spectra_summary.json"

# Canonical alias: summary.json (eval-data-contract expects data/spectra/summary.json)
if [[ -f "${SPECTRA_ROOT}/spectra_summary.json" ]] && [[ ! -e "${SPECTRA_ROOT}/summary.json" ]]; then
  ln -sfn "spectra_summary.json" "${SPECTRA_ROOT}/summary.json"
fi

echo "[INFO] Spectra computation complete"

# Clean up symlinks
rm -f "${WORK_LINK}"

if [[ "${WRITE_EXPERIMENT_CONFIG}" -eq 1 ]]; then
  cat > "${SPECTRA_ROOT}/EXPERIMENT_CONFIG.yaml" <<EOF
run_id: ${RUN_ID}
purpose: "ECMWF spectral-transform spectra from predictions"
pipeline:
  canonical_reference: ${PROJECT_ROOT}/eval/spectra/grb_to_spectra.sh
  stage_1: "predictions_*.nc -> nopoles GRIB (polar trimmed)"
  stage_2: "gptosp.ser (gridpoint -> spectral harmonics)"
  stage_3: "compute_spectra-3.py (spectral harmonics -> spectra amplitudes)"
  predictions:
    source_dir: ${PREDICTIONS_DIR}
    date_list: ${DATE_LIST}
    step_list: ${STEP_LIST}
    member_list: ${MEMBER_LIST}
    weather_states: ${WEATHER_STATES}
    template_root: ${TEMPLATE_ROOT}
    template_grib_root: ${TEMPLATE_GRIB_ROOT}
output:
  grb_dir: ${GRB_DIR}
  sh_dir: ${SH_DIR}
  spectra_dir: ${SPECTRA_DIR}
EOF
fi

echo "DONE: spectra in ${SPECTRA_DIR}"
