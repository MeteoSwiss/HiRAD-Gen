#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
# DEPRECATED (2026-05-07): use submit_scoreboard_chain.sh and the four
# scoreboard_{spectra,tc,surface,finalize}_step.sbatch templates instead.
# This monolith is preserved for one release and will be archived after
# downstream callers have been migrated.
# Canonical post-inference scoreboard writer (Pillars 2-4 + scoreboard refresh).
#
# Typical usage (after inference submit):
#   sbatch --dependency=afterok:<INFERENCE_JOB_ID> scoreboard_write_from_predictions.sbatch
#
# Or run immediately when predictions already exist:
#   sbatch scoreboard_write_from_predictions.sbatch
#
# This template:
#   1) runs spectra on the canonical proxy10 subset (10 fixed date/step pairs; proxy or trusted ECMWF),
#   2) runs TC Idalia+Franklin from predictions,
#   3) runs weighted surface-loss summary,
#   4) regenerates clean/global scoreboards + ENFO O320 scoreboard.
#
# Edit only USER SETTINGS.
#
#SBATCH --job-name=scoreboard_write
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

# If blank, RUN_ROOT defaults to "${OUTPUT_ROOT}/${RUN_ID}".
RUN_ROOT=""

# If blank, PRED_DIR defaults to "${RUN_ROOT}/predictions".
PRED_DIR=""
EXPECTED_PRED_FILES=25

# Sigma CSV run id, usually full checkpoint UUID used by sigma launcher.
# Keep blank to skip sigma lookup in the optional metrics JSON.
SIGMA_RUN_ID=""

# TC settings
TC_EVENTS="idalia,franklin"
TC_SUPPORT_MODE="regridded"       # native | regridded
BASE_TC_DIR="/home/ecm5702/perm/reference/o96_o320/tc"
TC_EXTRA_REFERENCE_EXPIDS=""      # comma-separated extra reference expids for TC PDF
TC_ANALYSIS_EXPID=""              # analysis truth anchor, e.g. OPER_O320_0001
TC_DISPLAY_LABEL=""               # blank => auto-cut from RUN_ID for TC PDF legends

# Spectra settings
# The canonical full-scoreboard spectra contract uses the same proxy10 subset as the
# fast gate: 10 fixed date/step pairs rather than the historical step120-only slice.
# SPECTRA_METHOD:
# - proxy  -> fast healpy/fft proxy route
# - ecmwf  -> trusted gp2sp route (AC only; uses gptosp.ser + compute_reference_spectra.py)
SPECTRA_METHOD="ecmwf"             # proxy | ecmwf
SPECTRA_BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"
SPECTRA_WEATHER_STATES_PROXY="10u,10v,2t,msl,t_850,z_500"
SPECTRA_WEATHER_STATES_ECMWF="10u,10v,2t,sp,t_850,z_500"
SPECTRA_NSIDE=128
SPECTRA_LMAX=319
SPECTRA_MEMBER_AGG="raw-members"   # per-file-mean | raw-members
SPECTRA_MEMBER_LIST="1,2,3,4,5,6,7,8,9,10"
SPECTRA_TEMPLATE_ROOT=""           # blank => auto-detect from prediction grid point count
SPECTRA_REFERENCE_BASE="/home/ecm5702/perm/reference/spectra"
SPECTRA_COMPARE_MODELS="enfo_o320,eefo_o96"
SPECTRA_REFERENCE_ROOT="${SPECTRA_REFERENCE_BASE}/enfo_o320"

# Scoreboard outputs
SCOREBOARD_MD="/home/ecm5702/scratch/eval/scoreboards/enfo_o320_scoreboard.md"
WRITE_CLEAN_SCOREBOARDS=1          # 1=yes, 0=no

# Optional consolidated metrics sidecar in run root
WRITE_METRICS_JSON=1               # 1=yes, 0=no
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
require_choice "${SPECTRA_METHOD}" proxy ecmwf

if [[ -z "${RUN_ROOT}" ]]; then
  RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
fi
if [[ -z "${PRED_DIR}" ]]; then
  PRED_DIR="${RUN_ROOT}/predictions"
fi

[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT not found: ${RUN_ROOT}"
[[ -d "${PRED_DIR}" ]] || die "PRED_DIR not found: ${PRED_DIR}"

pred_count=$(find "${PRED_DIR}" -maxdepth 1 -name 'predictions_*.nc' | wc -l)
[[ "${pred_count}" -eq "${EXPECTED_PRED_FILES}" ]] || die \
  "Expected ${EXPECTED_PRED_FILES} prediction files, found ${pred_count} under ${PRED_DIR}"

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) VENV_ACTIVATE="/home/ecm5702/dev/.ds-dyn/bin/activate" ;;
  ag*) VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag/bin/activate" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

CANONICAL_REPO_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
REPO_ROOT="${DOWNSCALING_TOOLS_ROOT:-${CANONICAL_REPO_ROOT}}"
TEMPLATE_DIR="${REPO_ROOT}/eval/jobs/templates"
[[ -d "${REPO_ROOT}" ]] || die "Repo root not found: ${REPO_ROOT}"
[[ -d "${TEMPLATE_DIR}" ]] || die "Template dir not found: ${TEMPLATE_DIR}"

source "${VENV_ACTIVATE}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi
PYTHONPATH_WITH_REPO="${PYTHONPATH}"
module load ecmwf-toolbox
METVIEW_BIN="$(command -v metview || true)"

if [[ "${SPECTRA_METHOD}" == "ecmwf" && "${HOST_SHORT}" != ac* ]]; then
  die "SPECTRA_METHOD=ecmwf requires AC login nodes; current host=${HOST_SHORT}"
fi

if [[ "${TC_SUPPORT_MODE}" == "regridded" ]]; then
  python -c "import metview" >/dev/null 2>&1 || \
    die "TC_SUPPORT_MODE=regridded requires metview. Use AC or switch to native mode."
fi

TC_EVENT_TAG="${TC_EVENTS//,/__}"
TC_EVENT_TAG="${TC_EVENT_TAG// /}"
if [[ -z "${TC_EVENT_TAG}" ]]; then
  TC_EVENT_TAG="all_events"
fi
# Canonical output paths (eval-data-contract)
mkdir -p "${RUN_ROOT}/data/tc" "${RUN_ROOT}/plots/tc" "${RUN_ROOT}/data/spectra"
TC_OUT_NAME="${TC_EVENT_TAG}.pdf"
TC_STATS="${RUN_ROOT}/data/tc/${TC_EVENT_TAG}.stats.json"
TC_PDF="${RUN_ROOT}/plots/tc/${TC_EVENT_TAG}.pdf"

# Spectra output — always canonical data/spectra
SPECTRA_OUT="${RUN_ROOT}/data/spectra"
# Legacy symlink for old-style paths
if [[ "${SPECTRA_METHOD}" == "ecmwf" ]]; then
  ln -sfn "data/spectra" "${RUN_ROOT}/spectra_proxy10_subset_ecmwf" 2>/dev/null || true
else
  ln -sfn "data/spectra" "${RUN_ROOT}/spectra_proxy10_subset_proxy" 2>/dev/null || true
fi
SUBSET_DIR="${RUN_ROOT}/predictions_proxy10_subset"
# Canonical surface loss and metrics paths
SFC_OUT="${RUN_ROOT}/data/surface_loss.json"
METRICS_JSON="${RUN_ROOT}/metrics.json"
mkdir -p "${RUN_ROOT}/data"
STAGE_HELPER="${TEMPLATE_DIR}/stage_prediction_spectra_gribs.py"
COMPUTE_REFERENCE="/home/ecm5702/dev/post_prepml/spectra/compute_reference_spectra.py"
PLOT_REFERENCE="${REPO_ROOT}/eval/spectra/plot_reference_models_compare.py"
PROXY_SPECTRA_SCRIPT="${TEMPLATE_DIR}/predictions_dir_spectra.py"
ECMWF_PARAM_DIRS=(2t_sfc 10u_sfc 10v_sfc sp_sfc t_850 z_500)

echo "=== Scoreboard writer ==="
echo "job_id=${SLURM_JOB_ID:-N/A}"
echo "host=${HOST_SHORT}"
echo "run_id=${RUN_ID}"
echo "run_root=${RUN_ROOT}"
echo "pred_dir=${PRED_DIR}"
echo "pred_count=${pred_count}"
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${SUBSET_DIR}"
IFS=',' read -r -a SPECTRA_PAIR_ARRAY <<< "${SPECTRA_BUNDLE_PAIRS}"
[[ "${#SPECTRA_PAIR_ARRAY[@]}" -gt 0 ]] || die "SPECTRA_BUNDLE_PAIRS must include at least one DATE:STEP pair."

subset_dates=()
subset_steps=()
for PAIR in "${SPECTRA_PAIR_ARRAY[@]}"; do
  [[ "${PAIR}" == *:* ]] || die "Invalid SPECTRA_BUNDLE_PAIRS entry: ${PAIR}"
  DATE="${PAIR%%:*}"
  STEP="${PAIR##*:}"
  [[ "${DATE}" =~ ^[0-9]{8}$ ]] || die "Invalid spectra date in pair: ${PAIR}"
  [[ "${STEP}" =~ ^[0-9]{1,3}$ ]] || die "Invalid spectra step in pair: ${PAIR}"
  printf -v STEP_PADDED "%03d" "${STEP}"
  src="${PRED_DIR}/predictions_${DATE}_step${STEP_PADDED}.nc"
  [[ -f "${src}" ]] || die "Missing spectra input file: ${src}"
  ln -sfn "${src}" "${SUBSET_DIR}/predictions_${DATE}_step${STEP_PADDED}.nc"
  subset_dates+=("${DATE}")
  subset_steps+=("${STEP}")
done

subset_count=$(find -L "${SUBSET_DIR}" -maxdepth 1 -type f -name 'predictions_*.nc' | wc -l)
[[ "${subset_count}" -eq "${#SPECTRA_PAIR_ARRAY[@]}" ]] || die \
  "Expected ${#SPECTRA_PAIR_ARRAY[@]} spectra subset files, found ${subset_count} under ${SUBSET_DIR}"

SPECTRA_DATE_START="$(printf '%s\n' "${subset_dates[@]}" | sort | head -n 1)"
SPECTRA_DATE_END="$(printf '%s\n' "${subset_dates[@]}" | sort | tail -n 1)"
SPECTRA_STEPS="$(printf '%s\n' "${subset_steps[@]}" | sort -n | uniq | paste -sd, -)"
SPECTRA_DATE_START_FMT="${SPECTRA_DATE_START:0:4}-${SPECTRA_DATE_START:4:2}-${SPECTRA_DATE_START:6:2}"
SPECTRA_DATE_END_FMT="${SPECTRA_DATE_END:0:4}-${SPECTRA_DATE_END:4:2}-${SPECTRA_DATE_END:6:2}"

if [[ "${SPECTRA_METHOD}" == "proxy" ]]; then
  python "${PROXY_SPECTRA_SCRIPT}" \
    --predictions-dir "${SUBSET_DIR}" \
    --out-dir "${SPECTRA_OUT}" \
    --run-label "${RUN_ID}_proxy10_subset" \
    --weather-states "${SPECTRA_WEATHER_STATES_PROXY}" \
    --nside "${SPECTRA_NSIDE}" \
    --lmax "${SPECTRA_LMAX}" \
    --member-aggregation "${SPECTRA_MEMBER_AGG}"
else
  [[ -f "${STAGE_HELPER}" ]] || die "Missing stage helper: ${STAGE_HELPER}"
  [[ -f "${COMPUTE_REFERENCE}" ]] || die "Missing compute script: ${COMPUTE_REFERENCE}"
  [[ -f "${PLOT_REFERENCE}" ]] || die "Missing spectra plot script: ${PLOT_REFERENCE}"

  if [[ -z "${SPECTRA_TEMPLATE_ROOT}" ]]; then
    first_subset="$(find -L "${SUBSET_DIR}" -maxdepth 1 -type f -name 'predictions_*.nc' | sort | head -n 1)"
    [[ -n "${first_subset}" ]] || die "No subset prediction files found in ${SUBSET_DIR}"
    SPECTRA_TEMPLATE_ROOT="$(python - "${first_subset}" <<'PY'
import sys
import xarray as xr
from pathlib import Path
p = Path(sys.argv[1])
with xr.open_dataset(p) as ds:
    n = int(ds["lat_hres"].size)
if n >= 6_000_000:
    print("/home/ecm5702/perm/reference/o320_o1280/spectra/enfo_o1280")
elif n >= 300_000:
    print("/home/ecm5702/perm/reference/o96_o320/spectra/enfo_o320")
else:
    print("/home/ecm5702/perm/reference/o48_o96/spectra/eefo_o96")
PY
)"
  fi
  [[ -d "${SPECTRA_TEMPLATE_ROOT}" ]] || die "Missing SPECTRA_TEMPLATE_ROOT: ${SPECTRA_TEMPLATE_ROOT}"
  [[ "${SPECTRA_LMAX}" =~ ^[0-9]+$ ]] || die "SPECTRA_LMAX must be an integer, got: ${SPECTRA_LMAX}"
  template_name="$(basename "${SPECTRA_TEMPLATE_ROOT}")"
  case "${template_name}" in
    enfo_o320)
      [[ "${SPECTRA_LMAX}" -eq 319 ]] || die "SPECTRA_LMAX=${SPECTRA_LMAX} incompatible with template ${SPECTRA_TEMPLATE_ROOT}; expected 319"
      [[ "${SPECTRA_REFERENCE_ROOT}" == "${SPECTRA_REFERENCE_BASE}/enfo_o320" ]] || die "SPECTRA_REFERENCE_ROOT must be ${SPECTRA_REFERENCE_BASE}/enfo_o320 for template ${SPECTRA_TEMPLATE_ROOT}"
      ;;
    enfo_o1280)
      [[ "${SPECTRA_LMAX}" -eq 1279 ]] || die "SPECTRA_LMAX=${SPECTRA_LMAX} incompatible with template ${SPECTRA_TEMPLATE_ROOT}; expected 1279"
      [[ "${SPECTRA_REFERENCE_ROOT}" == "${SPECTRA_REFERENCE_BASE}/enfo_o1280" ]] || die "SPECTRA_REFERENCE_ROOT must be ${SPECTRA_REFERENCE_BASE}/enfo_o1280 for template ${SPECTRA_TEMPLATE_ROOT}"
      ;;
  esac

  python "${STAGE_HELPER}" \
    --predictions-dir "${SUBSET_DIR}" \
    --out-dir "${SPECTRA_OUT}" \
    --template-root "${SPECTRA_TEMPLATE_ROOT}" \
    --weather-states "${SPECTRA_WEATHER_STATES_ECMWF}" \
    --summary-path "${SPECTRA_OUT}/staging_summary.json"

  module unload ecmwf-toolbox || true
  module load eclib
  module load pifsenv
  module load ifs
  export DR_HOOK_ASSERT_MPI_INITIALIZED=0
  SHORT_OUT="${TMPDIR}/scoreboard_sp_${SLURM_JOB_ID:-$$}"
  rm -f "${SHORT_OUT}"
  ln -s "${SPECTRA_OUT}" "${SHORT_OUT}"

  for param_dir in "${ECMWF_PARAM_DIRS[@]}"; do
    for grb in "${SHORT_OUT}/${param_dir}"/1_*_nopoles.grb; do
      [[ -f "${grb}" ]] || continue
      [[ -f "${grb}_sh" ]] && continue
      gptosp.ser -l -g "${grb}" -S "${grb}_sh"
    done
  done

  module unload ifs || true
  module unload pifsenv || true
  module unload eclib || true
  module load ecmwf-toolbox
  source "${VENV_ACTIVATE}"
  export PYTHONPATH="${PYTHONPATH_WITH_REPO}"
  if [[ -n "${METVIEW_BIN}" ]]; then
    export PATH="$(dirname "${METVIEW_BIN}"):${PATH}"
  fi

  python "${COMPUTE_REFERENCE}" \
    --in-dir "${SPECTRA_OUT}" \
    --out-dir "${SPECTRA_OUT}" \
    --expver-token 1 \
    --date-start "${SPECTRA_DATE_START_FMT}" \
    --date-end "${SPECTRA_DATE_END_FMT}" \
    --steps "${SPECTRA_STEPS}" \
    --numbers "${SPECTRA_MEMBER_LIST}" \
    --trunc "${SPECTRA_LMAX}"

  python - "${SPECTRA_OUT}" "${SPECTRA_REFERENCE_ROOT}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from eval.jobs import scoreboard_metrics as metrics

out_dir = Path(sys.argv[1])
ref_root = Path(sys.argv[2])
summary = metrics.build_spectra_summary(out_dir, reference_root=ref_root)
(out_dir / "spectra_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote scoreboard-compatible spectra summary: {out_dir / 'spectra_summary.json'}")
PY

  # Create canonical summary.json alias
  if [[ -f "${SPECTRA_OUT}/spectra_summary.json" ]] && [[ ! -e "${SPECTRA_OUT}/summary.json" ]]; then
    ln -sfn "spectra_summary.json" "${SPECTRA_OUT}/summary.json"
  fi

  IFS=',' read -r -a SPECTRA_COMPARE_ARRAY <<< "${SPECTRA_COMPARE_MODELS}"
  # For plot_reference, use a named symlink so base-dir model resolution works
  SPECTRA_PLOT_LINK="data_spectra"
  ln -sfn "${SPECTRA_OUT}" "${RUN_ROOT}/${SPECTRA_PLOT_LINK}" 2>/dev/null || true
  PLOT_MODELS="${SPECTRA_PLOT_LINK}"
  for ref_model in "${SPECTRA_COMPARE_ARRAY[@]}"; do
    [[ -n "${ref_model}" ]] || continue
    [[ "${ref_model}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe reference model token: ${ref_model}"
    ref_dir="${SPECTRA_REFERENCE_BASE}/${ref_model}"
    [[ -d "${ref_dir}" ]] || die "Missing spectra reference model dir: ${ref_dir}"
    ref_link="${RUN_ROOT}/${ref_model}"
    if [[ -e "${ref_link}" && ! -L "${ref_link}" ]]; then
      die "Run root path exists and is not a symlink: ${ref_link}"
    fi
    ln -sfn "${ref_dir}" "${ref_link}"
    PLOT_MODELS="${PLOT_MODELS},${ref_model}"
  done

  python "${PLOT_REFERENCE}" \
    --base-dir "${RUN_ROOT}" \
    --models "${PLOT_MODELS}" \
    --output-dir "${SPECTRA_OUT}" \
    --prefer-step "$(printf '%s\n' "${subset_steps[@]}" | sort -n | head -n 1)"
fi

cd "${REPO_ROOT}"
TC_CMD=(
  python -m eval._backends.tc.workflows pdf
  --predictions-dir "${PRED_DIR}"
  --outdir "${RUN_ROOT}/plots/tc"
  --run-label "${RUN_ID}"
  --out-name "${TC_OUT_NAME}"
  --base-tc-dir "${BASE_TC_DIR}"
  --support-mode "${TC_SUPPORT_MODE}"
)
if [[ -n "${TC_EVENTS}" ]]; then
  TC_CMD+=(--events "${TC_EVENTS}")
fi
if [[ -n "${TC_EXTRA_REFERENCE_EXPIDS}" ]]; then
  TC_CMD+=(--extra-reference-expids "${TC_EXTRA_REFERENCE_EXPIDS}")
fi
if [[ -n "${TC_DISPLAY_LABEL}" ]]; then
  TC_CMD+=(--display-label "${TC_DISPLAY_LABEL}")
fi
if [[ -n "${TC_ANALYSIS_EXPID}" ]]; then
  TC_CMD+=(--analysis-expid "${TC_ANALYSIS_EXPID}")
fi
"${TC_CMD[@]}"

# Move stats.json from plots/tc/ to data/tc/ (Python writes it alongside PDF)
TC_STATS_FROM_WORKFLOW="${RUN_ROOT}/plots/tc/${TC_EVENT_TAG}.stats.json"
if [[ -f "${TC_STATS_FROM_WORKFLOW}" ]] && [[ "${TC_STATS_FROM_WORKFLOW}" != "${TC_STATS}" ]]; then
  mv "${TC_STATS_FROM_WORKFLOW}" "${TC_STATS}"
fi

# Legacy compat symlinks
LEGACY_TC_STEM="tc_normed_pdfs_${TC_EVENT_TAG}_${RUN_ID}_from_predictions"
ln -sfn "data/tc/${TC_EVENT_TAG}.stats.json" "${RUN_ROOT}/${LEGACY_TC_STEM}.stats.json" 2>/dev/null || true
ln -sfn "plots/tc/${TC_EVENT_TAG}.pdf" "${RUN_ROOT}/${LEGACY_TC_STEM}.pdf" 2>/dev/null || true

python "${REPO_ROOT}/eval/jobs/scoreboard_surface_loss.py" \
  --predictions-dir "${PRED_DIR}" \
  --out-json "${SFC_OUT}"

# Legacy compat symlink for surface loss
ln -sfn "data/surface_loss.json" "${RUN_ROOT}/surface_loss_summary.json" 2>/dev/null || true

if [[ "${WRITE_CLEAN_SCOREBOARDS}" -eq 1 ]]; then
  python "${REPO_ROOT}/eval/jobs/generate_clean_scoreboards.py"
fi

python "${REPO_ROOT}/eval/jobs/generate_enfo_o320_scoreboard.py" \
  --eval-root "${OUTPUT_ROOT}" \
  --out-md "${SCOREBOARD_MD}"

if [[ "${WRITE_METRICS_JSON}" -eq 1 ]]; then
python - "${RUN_ID}" "${OUTPUT_ROOT}" "${SIGMA_RUN_ID}" "${TC_STATS}" "${SPECTRA_OUT}" "${SFC_OUT}" "${METRICS_JSON}" "${REPO_ROOT}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[8])
from eval.jobs import scoreboard_metrics as metrics

run_id = sys.argv[1]
output_root = Path(sys.argv[2])
sigma_run_id = sys.argv[3].strip()
tc_stats = Path(sys.argv[4])
spectra_dir = Path(sys.argv[5])
surface_json = Path(sys.argv[6])
out_json = Path(sys.argv[7])
run_root = Path(sys.argv[9])

# Try to read lane/checkpoint info from EXPERIMENT_CONFIG.yaml
lane = ""
checkpoint_id = ""
checkpoint_step = 0
scope = {}
config_path = run_root / "EXPERIMENT_CONFIG.yaml"
if config_path.exists():
    import yaml
    config = yaml.safe_load(config_path.read_text()) or {}
    lane = config.get("lane", "o96_o320")
    checkpoint_id = config.get("checkpoint_id", "")
    checkpoint_step = int(config.get("checkpoint_step", 0))
    scope = config.get("scope", {})

out = metrics.build_run_scoreboard_metrics(
    run_id=run_id,
    output_root=output_root,
    sigma_run_id=sigma_run_id,
    tc_stats_path=tc_stats,
    spectra_dir=spectra_dir,
    surface_json_path=surface_json,
    lane=lane,
    checkpoint_id=checkpoint_id,
    checkpoint_step=checkpoint_step,
    scope=scope,
)

out_json.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(f"Saved: {out_json}")
PY

# Legacy compat symlinks for metrics
ln -sfn "metrics.json" "${RUN_ROOT}/scoreboard_metrics.json" 2>/dev/null || true
fi

echo "DONE run_id=${RUN_ID}"
echo "SCOREBOARD_MD=${SCOREBOARD_MD}"
echo "TC_STATS=${TC_STATS}"
echo "TC_PDF=${TC_PDF}"
echo "SPECTRA_SUMMARY=${SPECTRA_OUT}/spectra_summary.json"
echo "SFC_SUMMARY=${SFC_OUT}"
[[ "${WRITE_METRICS_JSON}" -eq 1 ]] && echo "METRICS_JSON=${METRICS_JSON}"
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
