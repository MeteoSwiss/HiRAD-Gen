#!/bin/bash
# preflight_eval_check.sh — Sourceable pre-submission validation for eval jobs.
#
# Usage (sourced):
#   source /path/to/downscaling-tools/eval/jobs/templates/preflight_eval_check.sh
#   preflight_cluster           # sets PREFLIGHT_CLUSTER to ac|ag
#   preflight_venv "new"        # validates venv for stack flavor
#   preflight_predictions_dir "/path/to/predictions" 25   # checks file count
#   preflight_tc_refs "idalia"  # checks TC reference data exists
#   preflight_summary           # prints collected warnings/info
#
# Usage (standalone):
#   bash preflight_eval_check.sh --predictions-dir /path --stack new --lane o96_o320
#
# All functions set PREFLIGHT_WARNINGS (array) and PREFLIGHT_ERRORS (array).
# Call preflight_summary to print them. If PREFLIGHT_ERRORS is non-empty,
# preflight_summary exits with code 1.

PREFLIGHT_WARNINGS=()
PREFLIGHT_ERRORS=()
PREFLIGHT_CLUSTER=""
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${TEMPLATE_DIR}/../../.." && pwd)"
TC_REQUEST_SCRIPT="${PROJECT_ROOT}/eval/tc/all_events_request.sh"

preflight_cluster() {
  local host_short
  host_short="$(hostname -s)"
  case "${host_short}" in
    ac*) PREFLIGHT_CLUSTER="ac" ;;
    ag*) PREFLIGHT_CLUSTER="ag" ;;
    *)   PREFLIGHT_CLUSTER="unknown"
         PREFLIGHT_WARNINGS+=("Unknown cluster from hostname '${host_short}'") ;;
  esac
  echo "[preflight] cluster=${PREFLIGHT_CLUSTER} (${host_short})"
}

preflight_venv() {
  local stack="${1:-new}"
  local cluster="${PREFLIGHT_CLUSTER:-unknown}"

  # Canonical venv map (matches README design invariants)
  local venv_ac_new="/home/ecm5702/dev/.ds-dyn/bin/activate"
  local venv_ag_new="/home/ecm5702/dev/.ds-ag/bin/activate"
  local venv_ac_old="/home/ecm5702/dev/.ds-old/bin/activate"
  local venv_ag_old="/home/ecm5702/dev/.ds-ag-old/bin/activate"

  local key="${cluster}_${stack}"
  local venv_path=""
  case "${key}" in
    ac_new) venv_path="${venv_ac_new}" ;;
    ag_new) venv_path="${venv_ag_new}" ;;
    ac_old) venv_path="${venv_ac_old}" ;;
    ag_old) venv_path="${venv_ag_old}" ;;
    *)      PREFLIGHT_ERRORS+=("Cannot resolve venv for cluster=${cluster} stack=${stack}")
            return 1 ;;
  esac

  if [[ -f "${venv_path}" ]]; then
    echo "[preflight] venv OK: ${venv_path}"
  else
    PREFLIGHT_ERRORS+=("Venv not found: ${venv_path}")
    return 1
  fi

  # Export for downstream use
  export PREFLIGHT_VENV="${venv_path}"
}

preflight_python_packages() {
  # Check that critical packages are importable.
  # Call AFTER activating the venv.
  local packages=("$@")
  if [[ ${#packages[@]} -eq 0 ]]; then
    packages=(earthkit.data xarray matplotlib)
  fi
  local python="${PREFLIGHT_PYTHON:-python}"
  for pkg in "${packages[@]}"; do
    if "${python}" -c "import ${pkg}" 2>/dev/null; then
      echo "[preflight] package OK: ${pkg}"
    else
      PREFLIGHT_ERRORS+=("Python package missing: ${pkg} (in $(which "${python}"))")
    fi
  done
}

preflight_predictions_dir() {
  local pred_dir="$1"
  local expected_count="${2:-0}"  # 0 = skip count check

  if [[ ! -d "${pred_dir}" ]]; then
    PREFLIGHT_ERRORS+=("Predictions directory does not exist: ${pred_dir}")
    return 1
  fi

  local actual_count
  actual_count=$(find "${pred_dir}" -maxdepth 1 -name 'predictions_*.nc' -type f 2>/dev/null | wc -l)
  echo "[preflight] predictions_dir: ${pred_dir} (${actual_count} files)"

  if [[ "${expected_count}" -gt 0 && "${actual_count}" -ne "${expected_count}" ]]; then
    PREFLIGHT_WARNINGS+=("Expected ${expected_count} prediction files, found ${actual_count} in ${pred_dir}")
  fi

  if [[ "${actual_count}" -eq 0 ]]; then
    PREFLIGHT_ERRORS+=("No predictions_*.nc files found in ${pred_dir}")
  fi
}

preflight_tc_refs() {
  # Check that TC reference GRIB data exists for given events.
  local base_tc_dir="${2:-/home/ecm5702/perm/reference/o96_o320/tc}"
  local events
  IFS=',' read -ra events <<< "$1"
  for event in "${events[@]}"; do
    local event_dir="${base_tc_dir}/${event}"
    if [[ -d "${event_dir}" ]]; then
      local grib_count
      grib_count=$(find "${event_dir}" -maxdepth 1 -name '*.grib' -type f 2>/dev/null | wc -l)
      echo "[preflight] TC refs OK: ${event} (${grib_count} GRIBs in ${event_dir})"
      if [[ "${grib_count}" -eq 0 ]]; then
        PREFLIGHT_WARNINGS+=("TC dir exists but no GRIBs: ${event_dir}. Request data via: ${TC_REQUEST_SCRIPT}")
      fi
    else
      PREFLIGHT_ERRORS+=("TC reference dir missing: ${event_dir}. Request data via: ${TC_REQUEST_SCRIPT}")
    fi
  done
}

preflight_spectra_modules() {
  # Check that AC-only spectra modules are loadable.
  local cluster="${PREFLIGHT_CLUSTER:-unknown}"
  if [[ "${cluster}" != "ac" ]]; then
    PREFLIGHT_ERRORS+=("ECMWF spectra (gptosp) requires AC cluster. Current: ${cluster}")
    return 1
  fi
  for mod in ifs eclib; do
    if module is-avail "${mod}" 2>/dev/null; then
      echo "[preflight] module OK: ${mod}"
    else
      PREFLIGHT_WARNINGS+=("Module '${mod}' not available; gptosp may fail")
    fi
  done
}

preflight_qos() {
  local requested_qos="$1"
  local cluster="${PREFLIGHT_CLUSTER:-unknown}"

  # Known QOS rules (empirical from task failures)
  case "${cluster}" in
    ac)
      case "${requested_qos}" in
        nf|np|ef) echo "[preflight] QOS OK: ${requested_qos} on AC" ;;
        ng)       PREFLIGHT_WARNINGS+=("QOS 'ng' on AC is GPU-oriented. For CPU-only TC eval, prefer submit_tc_eval_from_predictions.sh (ac_cpu_safe) or switch to qos=nf and drop the GPU request; ng + --gpus-per-node=0 can fail with QOSMinGRES.") ;;
        *)        PREFLIGHT_WARNINGS+=("Unknown QOS '${requested_qos}' for AC") ;;
      esac
      ;;
    ag)
      case "${requested_qos}" in
        ng)       echo "[preflight] QOS OK: ${requested_qos} on AG" ;;
        nf)       PREFLIGHT_ERRORS+=("QOS 'nf' is not available on AG. Use 'ng' with --gpus-per-node=0 for CPU-only.") ;;
        *)        PREFLIGHT_WARNINGS+=("Unknown QOS '${requested_qos}' for AG") ;;
      esac
      ;;
  esac
}

preflight_walltime_guidance() {
  local job_type="$1"        # predict25 | predict75 | tc_eval | spectra_ecmwf | spectra_proxy
  local file_count="${2:-0}"

  # Empirical baselines from checkpoint-eval-pipeline task history
  echo "[preflight] Walltime guidance for job_type=${job_type}:"
  case "${job_type}" in
    predict25)
      echo "  Baseline: 12:00:00 for 25 files on GPU"
      echo "  If O1280: consider 24:00:00"
      ;;
    predict75)
      echo "  Baseline: 24:00:00 minimum for 75 files on GPU"
      echo "  Observed: 61/75 files in 12h → use 48:00:00 for safety"
      ;;
    tc_eval)
      echo "  Baseline: 04:00:00 with 128G memory"
      echo "  O1280 five-date ten-member: may need 256G"
      ;;
    spectra_ecmwf)
      echo "  Baseline: 48:00:00 for full gptosp + compute"
      echo "  Observed: 195/300 transforms in ~9h → budget 24h+ for 300 transforms"
      ;;
    spectra_proxy)
      echo "  Baseline: 08:00:00 with 128G (healpy-based)"
      ;;
    *)
      echo "  No baseline available for '${job_type}'"
      ;;
  esac
}

preflight_summary() {
  echo ""
  echo "=============================="
  echo " PREFLIGHT SUMMARY"
  echo "=============================="

  if [[ ${#PREFLIGHT_WARNINGS[@]} -gt 0 ]]; then
    echo ""
    echo "WARNINGS (${#PREFLIGHT_WARNINGS[@]}):"
    for w in "${PREFLIGHT_WARNINGS[@]}"; do
      echo "  [WARN] ${w}"
    done
  fi

  if [[ ${#PREFLIGHT_ERRORS[@]} -gt 0 ]]; then
    echo ""
    echo "ERRORS (${#PREFLIGHT_ERRORS[@]}):"
    for e in "${PREFLIGHT_ERRORS[@]}"; do
      echo "  [ERROR] ${e}"
    done
    echo ""
    echo "RESULT: FAIL — fix errors before submitting."
    return 1
  fi

  echo ""
  echo "RESULT: PASS — ready to submit."
  return 0
}

# Standalone mode
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  PRED_DIR=""
  STACK="new"
  LANE=""
  EVENTS=""
  JOB_TYPE=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --predictions-dir) PRED_DIR="$2"; shift 2 ;;
      --stack)           STACK="$2"; shift 2 ;;
      --lane)            LANE="$2"; shift 2 ;;
      --events)          EVENTS="$2"; shift 2 ;;
      --job-type)        JOB_TYPE="$2"; shift 2 ;;
      --help|-h)
        echo "Usage: $0 [--predictions-dir DIR] [--stack new|old] [--lane LANE] [--events EVENTS] [--job-type TYPE]"
        exit 0
        ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done

  preflight_cluster
  preflight_venv "${STACK}"

  if [[ -n "${PRED_DIR}" ]]; then
    preflight_predictions_dir "${PRED_DIR}"
  fi

  if [[ -n "${EVENTS}" ]]; then
    preflight_tc_refs "${EVENTS}"
  fi

  if [[ -n "${JOB_TYPE}" ]]; then
    preflight_walltime_guidance "${JOB_TYPE}"
  fi

  preflight_summary
fi
