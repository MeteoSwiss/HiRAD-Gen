#!/bin/bash
# Login-node helper: render and submit the canonical `o1280 -> o2560` manual-eval flow.
# Preferred use:
#   CHECKPOINT_PATH=<ABSOLUTE_BASE_OR_INFERENCE_CKPT> \
#   bash /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o1280_o2560_manual_eval_flow.sh

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
CHECKPOINT_PATH="${CHECKPOINT_PATH:-REPLACE_CHECKPOINT_PATH}"
SOURCE_HPC="${SOURCE_HPC:-ag}"                        # ac | ag | leonardo | jupiter
SOURCE_INPUT_ROOT="${SOURCE_INPUT_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/docs/epics/o1280_o2560/in-progress/humberto_launch_20260422/compat_o1280_inputs}"
SOURCE_FORCING_ROOT="${SOURCE_FORCING_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/docs/epics/o1280_o2560/in-progress/humberto_launch_20260422/compat_o2560_forcings}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ecm5702/scratch/eval}"
PHASE="${PHASE:-proof}"                              # proof | full-only

RUN_DATE_UTC="${RUN_DATE_UTC:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-manual_eval}"
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-}"

PROOF_BUNDLE_PAIRS="${PROOF_BUNDLE_PAIRS:-20250926:120}"
FULL_BUNDLE_PAIRS="${FULL_BUNDLE_PAIRS:-20250926:24,20250926:48,20250926:72,20250926:96,20250926:120,20250927:24,20250927:48,20250927:72,20250927:96,20250927:120,20250928:24,20250928:48,20250928:72,20250928:96,20250928:120,20250929:24,20250929:48,20250929:72,20250929:96,20250929:120,20250930:24,20250930:48,20250930:72,20250930:96,20250930:120}"
BUNDLE_PAIRS="${BUNDLE_PAIRS:-}"                     # optional exact date:step override
MEMBERS="${MEMBERS:-1}"

RUN_LOCAL_PLOTS="${RUN_LOCAL_PLOTS:-1}"
LOCAL_PLOT_DATE="${LOCAL_PLOT_DATE:-}"
LOCAL_PLOT_EXPECTED_COUNT="${LOCAL_PLOT_EXPECTED_COUNT:-auto}"
LOCAL_PLOT_OUT_SUBDIR="${LOCAL_PLOT_OUT_SUBDIR:-destine_showcase_plots}"

# O2560 DestinE showcase regions (uniform 3°×4° crops for 4.4 km vs 9 km).
# Space-separated list of region names from O2560_SHOWCASE_REGIONS in plot_regions.py.
# Blank means render all showcase regions defined for the detected grid.
LOCAL_PLOT_REGIONS="${LOCAL_PLOT_REGIONS:-humberto_core java_bali_strait gbr_reef_tight amazon_manaus congo_river_delta rift_valley_tight alps_innsbruck tokyo_bay florida_keys crete_south}"

RUN_SPECTRA="${RUN_SPECTRA:-1}"
RUN_SURFACE_LOSS="${RUN_SURFACE_LOSS:-1}"
SPECTRA_METHOD="${SPECTRA_METHOD:-proxy}"            # auto | proxy | ecmwf
ALLOW_DEBUG_FALLBACK="${ALLOW_DEBUG_FALLBACK:-0}"

HOLD="${HOLD:-0}"
NO_SUBMIT="${NO_SUBMIT:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
PROFILE_JSON_PATH="${PROFILE_JSON_PATH:-}"
PROFILE_PYTHON="${PROFILE_PYTHON:-}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-}"
PREFLIGHT_JSON_PATH="${PREFLIGHT_JSON_PATH:-}"
O2560_PLOT_MEM="${O2560_PLOT_MEM:-256G}"

RUN_TC_PDF="${RUN_TC_PDF:-1}"
TC_EVENTS="${TC_EVENTS:-humberto}"
TC_SUPPORT_MODE="${TC_SUPPORT_MODE:-auto}"
TC_EXTRA_REFERENCE_EXPIDS="${TC_EXTRA_REFERENCE_EXPIDS:-ENFO_O1280_0001}"
TC_ANALYSIS_EXPID="${TC_ANALYSIS_EXPID:-OPER_O1280_0001}"
TC_IEKM_TARGET_GRIB_PATH="${TC_IEKM_TARGET_GRIB_PATH:-/home/ecm5702/hpcperm/data/input_data/destine_iekm_o2560_targets_humberto_20250926_20250930/iekm_o2560_iekm_date*_time0000_step24to120_sfc_y.grib}"
TC_IEKM_LABEL="${TC_IEKM_LABEL:-IEKM_O2560_TARGET}"
TC_PLOT_TITLE="${TC_PLOT_TITLE:-TC Humberto 2025-09 | o1280→o2560 | norm. PDFs (MSLP & 10m Wind)}"
O2560_TC_MEM="${O2560_TC_MEM:-128G}"

RUN_TC_MEMBER_MAPS="${RUN_TC_MEMBER_MAPS:-1}"
TC_MEMBER_MAPS_OUT_DIR="${TC_MEMBER_MAPS_OUT_DIR:-tc_member_maps}"
TC_MEMBER_MAPS_MEMBERS="${TC_MEMBER_MAPS_MEMBERS:-0,1,2,3,4}"
TC_MEMBER_MAPS_DATE="${TC_MEMBER_MAPS_DATE:-20250928}"
TC_MEMBER_MAPS_STEPS="${TC_MEMBER_MAPS_STEPS:-24}"
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
TEMPLATE_DIR="${PROJECT_ROOT}/eval/jobs/templates"
source "${TEMPLATE_DIR}/render_helpers.sh"

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

require_bool() {
  local value="$1"
  [[ "${value}" == "0" || "${value}" == "1" ]] || die "Expected 0 or 1, got '${value}'"
}

require_source_gribs_for_date() {
  local date="$1"
  local required=(
    "${SOURCE_INPUT_ROOT}/enfo_o1280_0001_date${date}_time0000_step006to120by006_input.grib"
    "${SOURCE_FORCING_ROOT}/destine_rd_fc_oper_i4ql_o2560_date${date}_time0000_step006to120by006_sfc.grib"
    "${SOURCE_FORCING_ROOT}/destine_rd_fc_oper_i4ql_o2560_date${date}_time0000_step006to120by006_y.grib"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || die "Missing required source GRIB: ${path}"
  done
}

require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${PHASE}" proof full-only
require_choice "${SPECTRA_METHOD}" auto proxy ecmwf
require_bool "${RUN_LOCAL_PLOTS}"
require_bool "${RUN_SPECTRA}"
require_bool "${RUN_SURFACE_LOSS}"
require_bool "${ALLOW_DEBUG_FALLBACK}"
require_bool "${HOLD}"
require_bool "${NO_SUBMIT}"
require_bool "${ALLOW_OVERWRITE}"
require_bool "${RUN_TC_PDF}"
require_bool "${RUN_TC_MEMBER_MAPS}"
require_choice "${TC_SUPPORT_MODE}" auto native regridded
[[ "${O2560_TC_MEM}" =~ ^[0-9]+[KMGTP]$ ]] || die "O2560_TC_MEM must look like 256G"
[[ "${CHECKPOINT_PATH}" != REPLACE_* ]] || die "Set CHECKPOINT_PATH."
[[ -d "${SOURCE_INPUT_ROOT}" ]] || die "SOURCE_INPUT_ROOT does not exist: ${SOURCE_INPUT_ROOT}"
[[ -d "${SOURCE_FORCING_ROOT}" ]] || die "SOURCE_FORCING_ROOT does not exist: ${SOURCE_FORCING_ROOT}"
[[ "${RUN_SUFFIX}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_SUFFIX: ${RUN_SUFFIX}"
[[ "${RUN_ID_OVERRIDE}" =~ ^[A-Za-z0-9._-]*$ ]] || die "Unsafe RUN_ID_OVERRIDE: ${RUN_ID_OVERRIDE}"
[[ "${O2560_PLOT_MEM}" =~ ^[0-9]+[KMGTP]$ ]] || die "O2560_PLOT_MEM must look like 256G"
if [[ "${LOCAL_PLOT_EXPECTED_COUNT}" != "auto" && ! "${LOCAL_PLOT_EXPECTED_COUNT}" =~ ^[0-9]+$ ]]; then
  die "LOCAL_PLOT_EXPECTED_COUNT must be 'auto' or a non-negative integer."
fi

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac"; EXPECTED_PYTHON="/home/ecm5702/dev/.ds-dyn/bin/python" ;;
  ag*) HOST_FAMILY="ag"; EXPECTED_PYTHON="/home/ecm5702/dev/.ds-ag/bin/python" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

if [[ "${TC_SUPPORT_MODE}" == "auto" ]]; then
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    RESOLVED_TC_SUPPORT_MODE="regridded"
  else
    RESOLVED_TC_SUPPORT_MODE="native"
  fi
else
  RESOLVED_TC_SUPPORT_MODE="${TC_SUPPORT_MODE}"
fi
if [[ "${RESOLVED_TC_SUPPORT_MODE}" == "regridded" && "${HOST_FAMILY}" != "ac" ]]; then
  die "TC_SUPPORT_MODE=regridded requires an ac login node."
fi

PROFILE_PYTHON="${PROFILE_PYTHON:-${EXPECTED_PYTHON}}"
PREFLIGHT_PYTHON="${PREFLIGHT_PYTHON:-${EXPECTED_PYTHON}}"
[[ -x "${PREFLIGHT_PYTHON}" ]] || die "PREFLIGHT_PYTHON is not executable: ${PREFLIGHT_PYTHON}"
if [[ -z "${PROFILE_JSON_PATH}" && ! -x "${PROFILE_PYTHON}" ]]; then
  die "PROFILE_PYTHON is not executable: ${PROFILE_PYTHON}"
fi
if [[ -z "${PROFILE_JSON_PATH}" && "${PROFILE_PYTHON}" != "${EXPECTED_PYTHON}" ]]; then
  die "Host/env mismatch for o1280->o2560: ${HOST_FAMILY} requires ${EXPECTED_PYTHON}, got ${PROFILE_PYTHON}"
fi

CHECKPOINT_PATH_RESOLVED="$(readlink -f "${CHECKPOINT_PATH}")"
[[ -f "${CHECKPOINT_PATH_RESOLVED}" ]] || die "Checkpoint does not exist: ${CHECKPOINT_PATH_RESOLVED}"
CKPT_DIR="$(dirname "${CHECKPOINT_PATH_RESOLVED}")"
CKPT_NAME="$(basename "${CHECKPOINT_PATH_RESOLVED}")"
if [[ "${CKPT_NAME}" == inference-* ]]; then
  INFERENCE_COMPANION="${CHECKPOINT_PATH_RESOLVED}"
  BASE_CHECKPOINT="${CKPT_DIR}/${CKPT_NAME#inference-}"
else
  BASE_CHECKPOINT="${CHECKPOINT_PATH_RESOLVED}"
  INFERENCE_COMPANION="${CKPT_DIR}/inference-${CKPT_NAME}"
fi
[[ -f "${BASE_CHECKPOINT}" ]] || die "Base checkpoint is missing: ${BASE_CHECKPOINT}"
[[ -f "${INFERENCE_COMPANION}" ]] || die "Inference companion is missing: ${INFERENCE_COMPANION}"

if [[ -n "${PROFILE_JSON_PATH}" ]]; then
  PROFILE_JSON="$(cat "${PROFILE_JSON_PATH}")"
else
  PROFILE_JSON="$(
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
      "${PROFILE_PYTHON}" -m eval.jobs.checkpoint_profile \
        --name-ckpt "${INFERENCE_COMPANION}" \
        --source-hpc "${SOURCE_HPC}" \
        --host-short "${HOST_SHORT}" \
        --expected-lane o1280_o2560 \
        --allow-inference-companion \
        --json
  )"
fi

mapfile -t PROFILE_FIELDS < <(
  python3 - "${PROFILE_JSON}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(sys.argv[1])
ckpt_path = Path(payload["checkpoint_path"]).resolve()
print(payload["stack_flavor"])
print(payload["lane"])
print(payload["host_family"])
print(payload["recommended_venv"])
print(ckpt_path.parent.name[:8])
PY
)

STACK_FLAVOR="${PROFILE_FIELDS[0]}"
LANE="${PROFILE_FIELDS[1]}"
PROFILE_HOST_FAMILY="${PROFILE_FIELDS[2]}"
RECOMMENDED_VENV="${PROFILE_FIELDS[3]}"
CHECKPOINT_SHORT="${PROFILE_FIELDS[4]}"
[[ "${STACK_FLAVOR}" == "new" ]] || die "Expected new-stack checkpoint, got ${STACK_FLAVOR}"
[[ "${LANE}" == "o1280_o2560" ]] || die "Expected o1280_o2560 checkpoint lane, got ${LANE}"
[[ "${PROFILE_HOST_FAMILY}" == "${HOST_FAMILY}" ]] || die "Checkpoint-profile host mismatch: expected ${HOST_FAMILY}, got ${PROFILE_HOST_FAMILY}"

if [[ -n "${RUN_ID_OVERRIDE}" ]]; then
  RUN_ID="${RUN_ID_OVERRIDE}"
else
  RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
  [[ -n "${RUN_SUFFIX}" ]] && RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
fi

EFFECTIVE_BUNDLE_PAIRS="${BUNDLE_PAIRS:-$([[ "${PHASE}" == "proof" ]] && printf '%s' "${PROOF_BUNDLE_PAIRS}" || printf '%s' "${FULL_BUNDLE_PAIRS}")}"
mapfile -t SCOPE_FIELDS < <(
  python3 - "${EFFECTIVE_BUNDLE_PAIRS}" "${MEMBERS}" "${LOCAL_PLOT_DATE}" "${LOCAL_PLOT_EXPECTED_COUNT}" <<'PY'
from collections import Counter
import sys

pairs_raw, members_raw, local_plot_date_raw, expected_raw = sys.argv[1:5]
pairs = []
seen = set()
for token in pairs_raw.split(","):
    token = token.strip()
    if not token:
        continue
    date_raw, step_raw = token.split(":", 1)
    key = (date_raw.strip(), str(int(step_raw.strip())))
    if key in seen:
        continue
    seen.add(key)
    pairs.append(key)
if not pairs:
    raise SystemExit("Resolved bundle pair list is empty.")
members = [str(int(token.strip())) for token in members_raw.split(",") if token.strip()]
if not members:
    raise SystemExit("Resolved member list is empty.")
dates = []
steps = []
for date_value, step_value in pairs:
    if date_value not in dates:
        dates.append(date_value)
    if step_value not in steps:
        steps.append(step_value)
per_date_counts = Counter(date_value for date_value, _ in pairs)
local_plot_date = local_plot_date_raw.strip() or dates[0]
expected_count = int(expected_raw) if expected_raw != "auto" else per_date_counts.get(local_plot_date, 0)
print(",".join(f"{date}:{step}" for date, step in pairs))
print(",".join(dates))
print(",".join(steps))
print(",".join(members))
print(dates[0])
print(steps[0])
print(local_plot_date)
print(str(expected_count if expected_count > 0 else len(steps)))
PY
)

NORMALIZED_BUNDLE_PAIRS="${SCOPE_FIELDS[0]}"
EFFECTIVE_DATES="${SCOPE_FIELDS[1]}"
EFFECTIVE_STEPS="${SCOPE_FIELDS[2]}"
EFFECTIVE_MEMBERS="${SCOPE_FIELDS[3]}"
FIRST_DATE="${SCOPE_FIELDS[4]}"
FIRST_STEP="${SCOPE_FIELDS[5]}"
RESOLVED_LOCAL_PLOT_DATE="${SCOPE_FIELDS[6]}"
RESOLVED_LOCAL_PLOT_EXPECTED_COUNT="${SCOPE_FIELDS[7]}"

IFS=',' read -r -a DATE_LIST <<< "${EFFECTIVE_DATES}"
for date in "${DATE_LIST[@]}"; do
  [[ -n "${date}" ]] || continue
  require_source_gribs_for_date "${date}"
done

RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
BUNDLE_DIR="${RUN_ROOT}/bundles_with_y"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"
SUBMIT_DIR="${SUBMIT_ROOT}/${RUN_DATE_UTC}"
mkdir -p "${RUN_ROOT}/logs" "${SUBMIT_DIR}"
printf '%s\n' "${PROFILE_JSON}" > "${RUN_ROOT}/checkpoint_profile.json"

if [[ -n "${PREFLIGHT_JSON_PATH}" ]]; then
  PREFLIGHT_JSON="$(cat "${PREFLIGHT_JSON_PATH}")"
else
  PREFLIGHT_JSON="$(
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
      "${PREFLIGHT_PYTHON}" -m eval.jobs.o1280_o2560_bundle_preflight \
        --lres-input-grib "${SOURCE_INPUT_ROOT}/enfo_o1280_0001_date${FIRST_DATE}_time0000_step006to120by006_input.grib" \
        --hres-forcing-grib "${SOURCE_FORCING_ROOT}/destine_rd_fc_oper_i4ql_o2560_date${FIRST_DATE}_time0000_step006to120by006_sfc.grib" \
        --target-grib "${SOURCE_FORCING_ROOT}/destine_rd_fc_oper_i4ql_o2560_date${FIRST_DATE}_time0000_step006to120by006_y.grib" \
        --step-hours "${FIRST_STEP}" \
        --member 1 \
        --json
  )"
fi
printf '%s\n' "${PREFLIGHT_JSON}" > "${RUN_ROOT}/bundle_preflight.json"

mapfile -t PREFLIGHT_FIELDS < <(
  python3 - "${PREFLIGHT_JSON}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
contract = payload["contract"]
print("1" if payload["strict_bundle_ready"] else "0")
print("1" if payload["proof_only_ready"] else "0")
print(payload["blocker_summary"])
print(contract["output_weather_state_mode"])
print(contract["output_weather_states_csv"])
print(contract["plot_weather_states_csv"])
print(contract["spectra_weather_states_csv"])
print(str(contract["num_gpus_per_model"]))
print("1" if contract["slim_output"] else "0")
PY
)

STRICT_BUNDLE_READY="${PREFLIGHT_FIELDS[0]}"
PROOF_ONLY_READY="${PREFLIGHT_FIELDS[1]}"
BLOCKER_SUMMARY="${PREFLIGHT_FIELDS[2]}"
OUTPUT_WEATHER_STATE_MODE="${PREFLIGHT_FIELDS[3]}"
OUTPUT_WEATHER_STATES="${PREFLIGHT_FIELDS[4]}"
PLOT_WEATHER_STATES="${PREFLIGHT_FIELDS[5]}"
SPECTRA_WEATHER_STATES="${PREFLIGHT_FIELDS[6]}"
NUM_GPUS_PER_MODEL="${PREFLIGHT_FIELDS[7]}"
SLIM_OUTPUT="${PREFLIGHT_FIELDS[8]}"
RESOLVED_SPECTRA_METHOD="$([[ "${SPECTRA_METHOD}" == "auto" ]] && printf 'proxy' || printf '%s' "${SPECTRA_METHOD}")"

TC_TEMPLATE="${TEMPLATE_DIR}/tc_eval_from_predictions.sbatch"
TC_MEMBER_MAPS_TEMPLATE="${TEMPLATE_DIR}/tc_member_maps_from_predictions.sbatch"

[[ "${RUN_TC_PDF}" -eq 0 ]] || [[ -f "${TC_TEMPLATE}" ]] || die "Missing: ${TC_TEMPLATE}"
[[ "${RUN_TC_MEMBER_MAPS}" -eq 0 ]] || [[ -f "${TC_MEMBER_MAPS_TEMPLATE}" ]] || die "Missing: ${TC_MEMBER_MAPS_TEMPLATE}"

BUILD_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_build_truth_bundles.sbatch"
PREDICT_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_predict.sbatch"
DEBUG_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_debug_dataloader.sbatch"
LOCAL_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_local_plots.sbatch"
SPECTRA_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_spectra.sbatch"
SURFACE_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_surface_loss.sbatch"
TC_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_tc_eval.sbatch"
TC_MEMBER_MAPS_SCRIPT=""
[[ "${RUN_TC_MEMBER_MAPS}" -eq 1 ]] && TC_MEMBER_MAPS_SCRIPT="${SUBMIT_DIR}/${RUN_ID}_tc_member_maps.sbatch"

OVERWRITE_CHECK_LIST=("${BUILD_SCRIPT}" "${PREDICT_SCRIPT}" "${DEBUG_SCRIPT}" "${LOCAL_SCRIPT}" "${SPECTRA_SCRIPT}" "${SURFACE_SCRIPT}")
[[ "${RUN_TC_PDF}" -eq 1 ]] && OVERWRITE_CHECK_LIST+=("${TC_SCRIPT}")
[[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]] && OVERWRITE_CHECK_LIST+=("${TC_MEMBER_MAPS_SCRIPT}")
for path in "${OVERWRITE_CHECK_LIST[@]}"; do
  if [[ -e "${path}" && "${ALLOW_OVERWRITE}" -ne 1 ]]; then
    die "Rendered script already exists: ${path}. Set ALLOW_OVERWRITE=1 to replace it."
  fi
done

if [[ "${STRICT_BUNDLE_READY}" == "1" ]]; then
  cp "${TEMPLATE_DIR}/build_o1280_o2560_truth_bundles.sbatch" "${BUILD_SCRIPT}"
  set_var "${BUILD_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
  set_var "${BUILD_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
  set_var "${BUILD_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
  set_var "${BUILD_SCRIPT}" SOURCE_INPUT_ROOT "${SOURCE_INPUT_ROOT}"
  set_var "${BUILD_SCRIPT}" SOURCE_FORCING_ROOT "${SOURCE_FORCING_ROOT}"
  set_var "${BUILD_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
  set_var "${BUILD_SCRIPT}" BUNDLE_DIR "${BUNDLE_DIR}"
  set_var "${BUILD_SCRIPT}" DATES "${EFFECTIVE_DATES}"
  set_var "${BUILD_SCRIPT}" STEPS "${EFFECTIVE_STEPS}"
  set_var "${BUILD_SCRIPT}" BUNDLE_PAIRS "${NORMALIZED_BUNDLE_PAIRS}"
  set_var "${BUILD_SCRIPT}" MEMBERS "${EFFECTIVE_MEMBERS}"
  set_sbatch_directive "${BUILD_SCRIPT}" job-name "o2560_bundle_${CHECKPOINT_SHORT}"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    set_sbatch_directive "${BUILD_SCRIPT}" qos "nf"
    drop_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node
  else
    set_sbatch_directive "${BUILD_SCRIPT}" qos "ng"
    set_sbatch_directive "${BUILD_SCRIPT}" gpus-per-node "0"
  fi

  cp "${TEMPLATE_DIR}/strict_manual_predict_x_bundle.sbatch" "${PREDICT_SCRIPT}"
  set_var "${PREDICT_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
  set_var "${PREDICT_SCRIPT}" LANE "${LANE}"
  set_var "${PREDICT_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
  set_var "${PREDICT_SCRIPT}" CHECKPOINT_REF "${BASE_CHECKPOINT}"
  set_var "${PREDICT_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
  set_var "${PREDICT_SCRIPT}" INPUT_ROOT "${BUNDLE_DIR}"
  set_var "${PREDICT_SCRIPT}" DATE_PRESET "default"
  set_var "${PREDICT_SCRIPT}" DATES "${EFFECTIVE_DATES}"
  set_var "${PREDICT_SCRIPT}" STEPS "${EFFECTIVE_STEPS}"
  set_var "${PREDICT_SCRIPT}" BUNDLE_PAIRS "${NORMALIZED_BUNDLE_PAIRS}"
  set_var "${PREDICT_SCRIPT}" MEMBERS "${EFFECTIVE_MEMBERS}"
  set_var "${PREDICT_SCRIPT}" OUTPUT_ROOT "${OUTPUT_ROOT}"
  set_var "${PREDICT_SCRIPT}" RUN_ID_OVERRIDE "${RUN_ID}"
  set_var "${PREDICT_SCRIPT}" RUN_SUFFIX ""
  set_var "${PREDICT_SCRIPT}" NUM_GPUS_PER_MODEL "${NUM_GPUS_PER_MODEL}"
  set_var "${PREDICT_SCRIPT}" OUTPUT_WEATHER_STATE_MODE "${OUTPUT_WEATHER_STATE_MODE}"
  set_var "${PREDICT_SCRIPT}" OUTPUT_WEATHER_STATES "${OUTPUT_WEATHER_STATES}"
  set_var "${PREDICT_SCRIPT}" SLIM_OUTPUT "${SLIM_OUTPUT}"
  set_var "${PREDICT_SCRIPT}" ALLOW_EXISTING_RUN_DIR "1"
  set_var "${PREDICT_SCRIPT}" ALLOW_REBUILT_BUNDLE_ROOT "1"
  set_sbatch_directive "${PREDICT_SCRIPT}" job-name "o2560_predict_${CHECKPOINT_SHORT}"
  set_sbatch_directive "${PREDICT_SCRIPT}" ntasks-per-node "${NUM_GPUS_PER_MODEL}"
  set_sbatch_directive "${PREDICT_SCRIPT}" cpus-per-task "8"
  set_sbatch_directive "${PREDICT_SCRIPT}" gpus-per-node "${NUM_GPUS_PER_MODEL}"
  set_sbatch_directive "${PREDICT_SCRIPT}" time "48:00:00"
  set_sbatch_directive "${PREDICT_SCRIPT}" mem "0"

  if [[ "${RUN_LOCAL_PLOTS}" -eq 1 ]]; then
    cp "${TEMPLATE_DIR}/local_plots_one_date_from_predictions.sbatch" "${LOCAL_SCRIPT}"
    set_var "${LOCAL_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
    set_var "${LOCAL_SCRIPT}" RUN_ID "${RUN_ID}"
    set_var "${LOCAL_SCRIPT}" DATE "${RESOLVED_LOCAL_PLOT_DATE}"
    set_var "${LOCAL_SCRIPT}" OUT_SUBDIR "${LOCAL_PLOT_OUT_SUBDIR}"
    set_var "${LOCAL_SCRIPT}" EXPECTED_COUNT "${RESOLVED_LOCAL_PLOT_EXPECTED_COUNT}"
    set_var "${LOCAL_SCRIPT}" WEATHER_STATES "${PLOT_WEATHER_STATES}"
    # Canonical o1280→o2560 local-plot workflow: suite mode renders all
    # O2560 showcase regions sequentially from the first step file.
    set_var "${LOCAL_SCRIPT}" RENDER_MODE "suite"
    set_var "${LOCAL_SCRIPT}" REGION_LIST "${LOCAL_PLOT_REGIONS}"
    set_sbatch_directive "${LOCAL_SCRIPT}" job-name "o2560_local_${CHECKPOINT_SHORT}"
    set_sbatch_directive "${LOCAL_SCRIPT}" mem "${O2560_PLOT_MEM}"
    if [[ "${HOST_FAMILY}" == "ac" ]]; then
      set_sbatch_directive "${LOCAL_SCRIPT}" qos "nf"
      drop_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node
    else
      set_sbatch_directive "${LOCAL_SCRIPT}" qos "ng"
      set_sbatch_directive "${LOCAL_SCRIPT}" gpus-per-node "0"
    fi
  fi

  if [[ "${RUN_SPECTRA}" -eq 1 ]]; then
    if [[ "${RESOLVED_SPECTRA_METHOD}" == "proxy" ]]; then
      cp "${TEMPLATE_DIR}/spectra_proxy_from_predictions.sbatch" "${SPECTRA_SCRIPT}"
      set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
      set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
      set_var "${SPECTRA_SCRIPT}" PREDICTIONS_SOURCE_DIR "${PREDICTIONS_DIR}"
      set_var "${SPECTRA_SCRIPT}" SUBSET_TAG "selected_scope"
      set_var "${SPECTRA_SCRIPT}" SUBSET_DIR "${RUN_ROOT}/predictions_subset_selected_scope"
      set_var "${SPECTRA_SCRIPT}" OUT_DIR "${RUN_ROOT}/spectra_proxy_selected_scope"
      set_var "${SPECTRA_SCRIPT}" DATE_LIST "${EFFECTIVE_DATES}"
      set_var "${SPECTRA_SCRIPT}" STEP_LIST "${EFFECTIVE_STEPS}"
      set_var "${SPECTRA_SCRIPT}" WEATHER_STATES "${SPECTRA_WEATHER_STATES}"
      set_var "${SPECTRA_SCRIPT}" NSIDE "1024"
      set_var "${SPECTRA_SCRIPT}" LMAX "2559"
      set_var "${SPECTRA_SCRIPT}" MEMBER_AGGREGATION "raw-members"
      set_sbatch_directive "${SPECTRA_SCRIPT}" job-name "o2560_spectra_${CHECKPOINT_SHORT}"
      if [[ "${HOST_FAMILY}" == "ac" ]]; then
        set_sbatch_directive "${SPECTRA_SCRIPT}" qos "nf"
        drop_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node
      else
        set_sbatch_directive "${SPECTRA_SCRIPT}" qos "ng"
        set_sbatch_directive "${SPECTRA_SCRIPT}" gpus-per-node "0"
      fi
    else
      cp "${TEMPLATE_DIR}/spectra_ecmwf_from_predictions.sbatch" "${SPECTRA_SCRIPT}"
      set_var "${SPECTRA_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
      set_var "${SPECTRA_SCRIPT}" RUN_ID "${RUN_ID}"
      set_var "${SPECTRA_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
      set_var "${SPECTRA_SCRIPT}" DATE_LIST "${EFFECTIVE_DATES}"
      set_var "${SPECTRA_SCRIPT}" STEP_LIST "${EFFECTIVE_STEPS}"
      set_var "${SPECTRA_SCRIPT}" MEMBER_LIST "ALL"
      set_var "${SPECTRA_SCRIPT}" WEATHER_STATES "${SPECTRA_WEATHER_STATES}"
      set_var "${SPECTRA_SCRIPT}" TEMPLATE_GRIB_ROOT "${SOURCE_FORCING_ROOT}"
      set_sbatch_directive "${SPECTRA_SCRIPT}" job-name "o2560_spectra_${CHECKPOINT_SHORT}"
    fi
  fi

  if [[ "${RUN_SURFACE_LOSS}" -eq 1 ]]; then
    cp "${TEMPLATE_DIR}/surface_loss_from_predictions.sbatch" "${SURFACE_SCRIPT}"
    set_var "${SURFACE_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
    set_var "${SURFACE_SCRIPT}" RUN_ID "${RUN_ID}"
    set_var "${SURFACE_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
    set_var "${SURFACE_SCRIPT}" OUT_JSON "${RUN_ROOT}/surface_loss_summary.json"
    set_sbatch_directive "${SURFACE_SCRIPT}" job-name "o2560_surface_${CHECKPOINT_SHORT}"
    if [[ "${HOST_FAMILY}" == "ac" ]]; then
      set_sbatch_directive "${SURFACE_SCRIPT}" qos "nf"
      drop_sbatch_directive "${SURFACE_SCRIPT}" gpus-per-node
    else
      set_sbatch_directive "${SURFACE_SCRIPT}" qos "ng"
      set_sbatch_directive "${SURFACE_SCRIPT}" gpus-per-node "0"
    fi
  fi

  if [[ "${RUN_TC_PDF}" -eq 1 ]]; then
    cp "${TC_TEMPLATE}" "${TC_SCRIPT}"
    set_var "${TC_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
    set_var "${TC_SCRIPT}" RUN_ID "${RUN_ID}"
    set_var "${TC_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
    set_var "${TC_SCRIPT}" EVENTS "${TC_EVENTS}"
    set_var "${TC_SCRIPT}" SUPPORT_MODE "${RESOLVED_TC_SUPPORT_MODE}"
    set_var "${TC_SCRIPT}" EXTRA_REFERENCE_EXPIDS "${TC_EXTRA_REFERENCE_EXPIDS}"
    set_var "${TC_SCRIPT}" ANALYSIS_EXPID "${TC_ANALYSIS_EXPID}"
    set_var "${TC_SCRIPT}" PLOT_TITLE "${TC_PLOT_TITLE}"
    set_var "${TC_SCRIPT}" IEKM_TARGET_GRIB_PATH "${TC_IEKM_TARGET_GRIB_PATH}"
    set_var "${TC_SCRIPT}" IEKM_LABEL "${TC_IEKM_LABEL}"
    set_sbatch_directive "${TC_SCRIPT}" job-name "o2560_tc_${CHECKPOINT_SHORT}"
    set_sbatch_directive "${TC_SCRIPT}" mem "${O2560_TC_MEM}"
    if [[ "${HOST_FAMILY}" == "ac" ]]; then
      set_sbatch_directive "${TC_SCRIPT}" qos "nf"
      drop_sbatch_directive "${TC_SCRIPT}" gpus-per-node
    else
      set_sbatch_directive "${TC_SCRIPT}" qos "ng"
      set_sbatch_directive "${TC_SCRIPT}" gpus-per-node "0"
    fi
  fi

  if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
    cp "${TC_MEMBER_MAPS_TEMPLATE}" "${TC_MEMBER_MAPS_SCRIPT}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_ID "${RUN_ID}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" PREDICTIONS_DIR "${PREDICTIONS_DIR}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" OUT_DIR "${RUN_ROOT}/${TC_MEMBER_MAPS_OUT_DIR}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" EVENTS "${TC_EVENTS}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" DATE "${TC_MEMBER_MAPS_DATE}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" STEPS "${TC_MEMBER_MAPS_STEPS}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" MEMBERS "${TC_MEMBER_MAPS_MEMBERS}"
    set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_LABEL "${CHECKPOINT_SHORT}"
    set_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" job-name "o2560_tcmbr_${CHECKPOINT_SHORT}"
    set_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" mem "${O2560_TC_MEM}"
    if [[ "${HOST_FAMILY}" == "ac" ]]; then
      set_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" qos "nf"
      drop_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" gpus-per-node
    else
      set_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" qos "ng"
      set_sbatch_directive "${TC_MEMBER_MAPS_SCRIPT}" gpus-per-node "0"
    fi
  fi
else
  [[ "${ALLOW_DEBUG_FALLBACK}" -eq 1 && "${PHASE}" == "proof" && "${PROOF_ONLY_READY}" == "1" ]] || die "Strict bundle route is not ready. Blockers: ${BLOCKER_SUMMARY}"
  cp "${TEMPLATE_DIR}/debug_from_dataloader_with_plots.sbatch" "${DEBUG_SCRIPT}"
  set_var "${DEBUG_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
  set_var "${DEBUG_SCRIPT}" LANE "${LANE}"
  set_var "${DEBUG_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
  set_var "${DEBUG_SCRIPT}" CHECKPOINT_REF "${BASE_CHECKPOINT}"
  set_var "${DEBUG_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
  set_var "${DEBUG_SCRIPT}" INFERENCE_COMPANION "${INFERENCE_COMPANION}"
  set_var "${DEBUG_SCRIPT}" USE_INFERENCE_COMPANION "true"
  set_var "${DEBUG_SCRIPT}" RUN_SUFFIX "${RUN_SUFFIX}_debug"
  set_var "${DEBUG_SCRIPT}" NTASKS "${NUM_GPUS_PER_MODEL}"
  set_var "${DEBUG_SCRIPT}" PLOT_WEATHER_STATES "${PLOT_WEATHER_STATES}"
  set_var "${DEBUG_SCRIPT}" DEBUG_ACK "I_UNDERSTAND_THIS_IS_DEBUG_ONLY"
  set_var "${DEBUG_SCRIPT}" EXTRA_PREDICT_FLAGS "--output-weather-state-mode all --output-weather-states ${OUTPUT_WEATHER_STATES} --slim-output"
  set_sbatch_directive "${DEBUG_SCRIPT}" job-name "o2560_debug_${CHECKPOINT_SHORT}"
  set_sbatch_directive "${DEBUG_SCRIPT}" ntasks-per-node "${NUM_GPUS_PER_MODEL}"
  set_sbatch_directive "${DEBUG_SCRIPT}" cpus-per-task "8"
  set_sbatch_directive "${DEBUG_SCRIPT}" gpus-per-node "${NUM_GPUS_PER_MODEL}"
fi

  if [[ "${NO_SUBMIT}" -eq 0 ]]; then
  SBATCH_ARGS=()
  [[ "${HOLD}" -eq 1 ]] && SBATCH_ARGS+=(--hold)
  if [[ "${STRICT_BUNDLE_READY}" == "1" ]]; then
    build_submit="$(sbatch "${SBATCH_ARGS[@]}" "${BUILD_SCRIPT}")"
    BUILD_JOB="$(extract_job_id "${build_submit}")"
    predict_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${BUILD_JOB} "${PREDICT_SCRIPT}")"
    PREDICT_JOB="$(extract_job_id "${predict_submit}")"
    [[ "${RUN_LOCAL_PLOTS}" -eq 1 ]] && LOCAL_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${PREDICT_JOB} "${LOCAL_SCRIPT}")")" || LOCAL_JOB=""
    [[ "${RUN_SPECTRA}" -eq 1 ]] && SPECTRA_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${PREDICT_JOB} "${SPECTRA_SCRIPT}")")" || SPECTRA_JOB=""
    [[ "${RUN_SURFACE_LOSS}" -eq 1 ]] && SURFACE_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${PREDICT_JOB} "${SURFACE_SCRIPT}")")" || SURFACE_JOB=""

    TC_JOB=""
    [[ "${RUN_TC_PDF}" -eq 1 ]] && TC_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${PREDICT_JOB} "${TC_SCRIPT}")")"

    TC_MEMBER_MAPS_JOB=""
    [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]] && TC_MEMBER_MAPS_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${PREDICT_JOB} "${TC_MEMBER_MAPS_SCRIPT}")")"

    # --- Finalize: lean eval layout reorganization ---
    FINALIZE_TEMPLATE="${TEMPLATE_DIR}/finalize_lean_eval_layout.sbatch"
    FINALIZE_SCRIPT="${SUBMIT_DIR}/finalize_lean_eval_${RUN_ID}.sbatch"
    finalize_job=""
    if [[ -f "${FINALIZE_TEMPLATE}" ]]; then
      if [[ ! -f "${FINALIZE_SCRIPT}" ]] || [[ "${ALLOW_OVERWRITE}" -eq 1 ]]; then
        cp "${FINALIZE_TEMPLATE}" "${FINALIZE_SCRIPT}"
        set_var "${FINALIZE_SCRIPT}" RUN_ROOT "${RUN_ROOT}"
        set_var "${FINALIZE_SCRIPT}" RUN_ID "${RUN_ID}"
        if [[ "${HOST_FAMILY}" == "ac" ]]; then
          set_sbatch_directive "${FINALIZE_SCRIPT}" qos "nf"
          drop_sbatch_directive "${FINALIZE_SCRIPT}" gpus-per-node
        else
          set_sbatch_directive "${FINALIZE_SCRIPT}" qos "ng"
          set_sbatch_directive "${FINALIZE_SCRIPT}" gpus-per-node "0"
        fi
      fi
      finalize_deps="${PREDICT_JOB}"
      [[ -n "${LOCAL_JOB}" ]] && finalize_deps+=":${LOCAL_JOB}"
      [[ -n "${SPECTRA_JOB}" ]] && finalize_deps+=":${SPECTRA_JOB}"
      [[ -n "${SURFACE_JOB}" ]] && finalize_deps+=":${SURFACE_JOB}"
      [[ -n "${TC_JOB}" ]] && finalize_deps+=":${TC_JOB}"
      [[ -n "${TC_MEMBER_MAPS_JOB}" ]] && finalize_deps+=":${TC_MEMBER_MAPS_JOB}"
      finalize_submit="$(sbatch "${SBATCH_ARGS[@]}" --dependency=afterok:${finalize_deps} "${FINALIZE_SCRIPT}")"
      finalize_job="$(extract_job_id "${finalize_submit}")"
      echo "[o2560-flow] ${finalize_submit}"
    fi
  else
    DEBUG_JOB="$(extract_job_id "$(sbatch "${SBATCH_ARGS[@]}" "${DEBUG_SCRIPT}")")"
  fi
fi

echo "[o2560-flow] run_id=${RUN_ID}"
echo "[o2560-flow] host_short=${HOST_SHORT}"
echo "[o2560-flow] recommended_venv=${RECOMMENDED_VENV}"
echo "[o2560-flow] base_checkpoint=${BASE_CHECKPOINT}"
echo "[o2560-flow] inference_companion=${INFERENCE_COMPANION}"
echo "[o2560-flow] run_root=${RUN_ROOT}"
echo "[o2560-flow] strict_bundle_ready=${STRICT_BUNDLE_READY}"
echo "[o2560-flow] blocker_summary=${BLOCKER_SUMMARY}"
echo "[o2560-flow] bundle_pairs=${NORMALIZED_BUNDLE_PAIRS}"
echo "[o2560-flow] members=${EFFECTIVE_MEMBERS}"
echo "[o2560-flow] output_weather_states=${OUTPUT_WEATHER_STATES}"
echo "[o2560-flow] spectra_method=${RESOLVED_SPECTRA_METHOD}"
echo "[o2560-flow] surface_loss=${RUN_SURFACE_LOSS}"
echo "[o2560-flow] run_tc_pdf=${RUN_TC_PDF}"
echo "[o2560-flow] tc_events=${TC_EVENTS}"
echo "[o2560-flow] tc_support_mode=${RESOLVED_TC_SUPPORT_MODE}"
echo "[o2560-flow] run_tc_member_maps=${RUN_TC_MEMBER_MAPS}"
echo "[o2560-flow] tc_job=${TC_JOB:-skipped}"
echo "[o2560-flow] tc_member_maps_job=${TC_MEMBER_MAPS_JOB:-skipped}"
echo "[o2560-flow] finalize_lean=${finalize_job:-skipped}"
echo "[o2560-flow] rendered_submit_dir=${SUBMIT_DIR}"
