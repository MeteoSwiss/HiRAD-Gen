#!/bin/bash
# Login-node helper: render and submit the canonical Aug 26-30 scoreboard flow.
#
# This helper creates run-specific sbatch launchers from templates and submits:
#   1) sigma evaluator (Pillar 1),
#   2) 25-file manual inference (Aug 26-30),
#   3) post-inference scoreboard writer (Pillars 2-4 + scoreboard refresh)
#      with dependency on both jobs above.
#
# Usage:
#   bash submit_aug26_30_scoreboard_flow.sh

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
CHECKPOINT_DIR="${CHECKPOINT_DIR:-REPLACE_CHECKPOINT_DIR}"      # e.g. 0c446b4118b94ec2bbec56c00409d664
CHECKPOINT_SHORT="${CHECKPOINT_SHORT:-REPLACE_CHECKPOINT_SHORT}" # e.g. 0c446b41
CKPT_NAME="${CKPT_NAME:-REPLACE_CHECKPOINT_FILENAME.ckpt}"      # e.g. anemoi-by_epoch-epoch_021-step_100000.ckpt

STACK_FLAVOR="${STACK_FLAVOR:-new}"             # new | old
LANE="${LANE:-o96_o320}"                        # o96_o320 | o320_o1280 | o1280_o2560
SOURCE_HPC="${SOURCE_HPC:-ac}"                  # ac | ag | leonardo | jupiter

# Run-id shape:
# manual_<CHECKPOINT_SHORT>_<STACK_FLAVOR>_<LANE>_<RUN_DATE_UTC>[_<RUN_SUFFIX>]
RUN_DATE_UTC="${RUN_DATE_UTC:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-scoreboard_aug26_30}"

# Post-inference writer settings
SPECTRA_METHOD="${SPECTRA_METHOD:-ecmwf}"       # proxy | ecmwf
TC_SUPPORT_MODE="${TC_SUPPORT_MODE:-regridded}" # native | regridded
TC_EVENTS="${TC_EVENTS:-idalia,franklin}"
TC_EXTRA_REFERENCE_EXPIDS="${TC_EXTRA_REFERENCE_EXPIDS:-ENFO_O320_0001,EEFO_O96_0001,ENFO_O320_ip6y}"
TC_ANALYSIS_EXPID="${TC_ANALYSIS_EXPID:-OPER_O320_0001}"

# TC member maps settings
RUN_TC_MEMBER_MAPS="${RUN_TC_MEMBER_MAPS:-1}"         # 1 => render per-member TC spatial maps
TC_MEMBER_MAPS_MEMBERS="${TC_MEMBER_MAPS_MEMBERS:-0,1,2,3,4}"
TC_MEMBER_MAPS_DATE="${TC_MEMBER_MAPS_DATE:-20230828}"  # strong-TC date for Idalia/Franklin
TC_MEMBER_MAPS_STEPS="${TC_MEMBER_MAPS_STEPS:-48}"      # single strong step (valid Aug 30: Idalia Cat3 + Franklin Cat4)

# Submission controls
SUBMIT_QOS="${SUBMIT_QOS:-}"                    # legacy override applied to all jobs when set
SUBMIT_QOS_GPU="${SUBMIT_QOS_GPU:-}"            # optional gpu-job qos override (sigma + inference)
SUBMIT_QOS_CPU="${SUBMIT_QOS_CPU:-}"            # optional cpu-job qos override (post-writer)
SUBMIT_HOLD="${SUBMIT_HOLD:-0}"                 # 1 => submit all jobs held (scheduler acceptance test)
SUBMIT_TIME_OVERRIDE="${SUBMIT_TIME_OVERRIDE:-}" # optional sbatch --time override (e.g. 00:10:00 for qos tests)
SUBMIT_GPUS_OVERRIDE="${SUBMIT_GPUS_OVERRIDE:-}" # optional sbatch --gpus-per-node override (e.g. 1 for qos requiring GRES)

# Where generated sbatch files are written
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"         # 1 => overwrite generated files if they exist
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

require_choice "${STACK_FLAVOR}" new old
require_choice "${LANE}" o96_o320 o320_o1280 o1280_o2560
require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
require_choice "${SPECTRA_METHOD}" proxy ecmwf
require_choice "${TC_SUPPORT_MODE}" native regridded
require_choice "${SUBMIT_HOLD}" 0 1
[[ "${CHECKPOINT_DIR}" != REPLACE_* ]] || die "Set CHECKPOINT_DIR."
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ "${CKPT_NAME}" != REPLACE_* ]] || die "Set CKPT_NAME."
if [[ -n "${SUBMIT_QOS}" && ! "${SUBMIT_QOS}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  die "Unsafe SUBMIT_QOS value: ${SUBMIT_QOS}"
fi
if [[ -n "${SUBMIT_QOS_GPU}" && ! "${SUBMIT_QOS_GPU}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  die "Unsafe SUBMIT_QOS_GPU value: ${SUBMIT_QOS_GPU}"
fi
if [[ -n "${SUBMIT_QOS_CPU}" && ! "${SUBMIT_QOS_CPU}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  die "Unsafe SUBMIT_QOS_CPU value: ${SUBMIT_QOS_CPU}"
fi
if [[ -n "${SUBMIT_TIME_OVERRIDE}" && ! "${SUBMIT_TIME_OVERRIDE}" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
  die "SUBMIT_TIME_OVERRIDE must use HH:MM:SS format."
fi
if [[ -n "${SUBMIT_GPUS_OVERRIDE}" && ! "${SUBMIT_GPUS_OVERRIDE}" =~ ^[0-9]+$ ]]; then
  die "SUBMIT_GPUS_OVERRIDE must be an integer >= 0."
fi

CHECKPOINT_REF="${CHECKPOINT_DIR}/${CKPT_NAME}"
RUN_ID="manual_${CHECKPOINT_SHORT}_${STACK_FLAVOR}_${LANE}_${RUN_DATE_UTC}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_ID="${RUN_ID}_${RUN_SUFFIX}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "Unsafe RUN_ID: ${RUN_ID}"

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEMPLATE_DIR}/render_helpers.sh"
INFER_TEMPLATE="${TEMPLATE_DIR}/strict_manual_predict_x_bundle.sbatch"
SIGMA_TEMPLATE="${TEMPLATE_DIR}/scoreboard_sigma_eval.sbatch"
WRITE_TEMPLATE="${TEMPLATE_DIR}/scoreboard_write_from_predictions.sbatch"

[[ -f "${INFER_TEMPLATE}" ]] || die "Missing template: ${INFER_TEMPLATE}"
[[ -f "${SIGMA_TEMPLATE}" ]] || die "Missing template: ${SIGMA_TEMPLATE}"
[[ -f "${WRITE_TEMPLATE}" ]] || die "Missing template: ${WRITE_TEMPLATE}"

TC_MEMBER_MAPS_TEMPLATE="${TEMPLATE_DIR}/tc_member_maps_from_predictions.sbatch"
if [[ "${RUN_TC_MEMBER_MAPS}" == "1" ]]; then
  [[ -f "${TC_MEMBER_MAPS_TEMPLATE}" ]] || die "Missing template: ${TC_MEMBER_MAPS_TEMPLATE}"
fi

SUBMIT_DIR="${SUBMIT_ROOT}/${RUN_DATE_UTC}"
mkdir -p "${SUBMIT_DIR}"

INFER_SCRIPT="${SUBMIT_DIR}/inference_${CHECKPOINT_SHORT}_${RUN_DATE_UTC}.sbatch"
SIGMA_SCRIPT="${SUBMIT_DIR}/sigma_${CHECKPOINT_SHORT}_${RUN_DATE_UTC}.sbatch"
WRITE_SCRIPT="${SUBMIT_DIR}/scoreboard_write_${CHECKPOINT_SHORT}_${RUN_DATE_UTC}.sbatch"

TC_MEMBER_MAPS_SCRIPT=""
if [[ "${RUN_TC_MEMBER_MAPS}" == "1" ]]; then
  TC_MEMBER_MAPS_SCRIPT="${SUBMIT_DIR}/tc_member_maps_${CHECKPOINT_SHORT}_${RUN_DATE_UTC}.sbatch"
fi

TARGET_LIST=("${INFER_SCRIPT}" "${SIGMA_SCRIPT}" "${WRITE_SCRIPT}")
if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
  TARGET_LIST+=("${TC_MEMBER_MAPS_SCRIPT}")
fi
for target in "${TARGET_LIST[@]}"; do
  if [[ -e "${target}" && "${ALLOW_OVERWRITE}" -ne 1 ]]; then
    die "Refusing to overwrite existing generated file: ${target} (set ALLOW_OVERWRITE=1 to replace)"
  fi
done

cp "${INFER_TEMPLATE}" "${INFER_SCRIPT}"
cp "${SIGMA_TEMPLATE}" "${SIGMA_SCRIPT}"
cp "${WRITE_TEMPLATE}" "${WRITE_SCRIPT}"

if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
  cp "${TC_MEMBER_MAPS_TEMPLATE}" "${TC_MEMBER_MAPS_SCRIPT}"
fi

# Render inference script (Aug 26-30 default preset).
set_var "${INFER_SCRIPT}" STACK_FLAVOR "${STACK_FLAVOR}"
set_var "${INFER_SCRIPT}" LANE "${LANE}"
set_var "${INFER_SCRIPT}" SOURCE_HPC "${SOURCE_HPC}"
set_var "${INFER_SCRIPT}" CHECKPOINT_REF "${CHECKPOINT_REF}"
set_var "${INFER_SCRIPT}" CHECKPOINT_SHORT "${CHECKPOINT_SHORT}"
set_var "${INFER_SCRIPT}" DATE_PRESET "default"
set_var "${INFER_SCRIPT}" DATES ""
set_var "${INFER_SCRIPT}" RUN_DATE_UTC "${RUN_DATE_UTC}"
set_var "${INFER_SCRIPT}" RUN_SUFFIX "${RUN_SUFFIX}"

# Render sigma script.
set_var "${SIGMA_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${SIGMA_SCRIPT}" CHECKPOINT_REF "${CHECKPOINT_DIR}"
set_var "${SIGMA_SCRIPT}" CKPT_NAME "${CKPT_NAME}"
set_var "${SIGMA_SCRIPT}" EXPECTED_LANE "${LANE}"

# Render post-inference writer.
set_var "${WRITE_SCRIPT}" RUN_ID "${RUN_ID}"
set_var "${WRITE_SCRIPT}" SIGMA_RUN_ID "${RUN_ID}"
set_var "${WRITE_SCRIPT}" SPECTRA_METHOD "${SPECTRA_METHOD}"
set_var "${WRITE_SCRIPT}" TC_SUPPORT_MODE "${TC_SUPPORT_MODE}"
set_var "${WRITE_SCRIPT}" TC_EVENTS "${TC_EVENTS}"
set_var "${WRITE_SCRIPT}" TC_EXTRA_REFERENCE_EXPIDS "${TC_EXTRA_REFERENCE_EXPIDS}"
set_var "${WRITE_SCRIPT}" TC_ANALYSIS_EXPID "${TC_ANALYSIS_EXPID}"

# Render TC member maps script.
if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
  AUG_RUN_ROOT="/home/ecm5702/scratch/eval/${RUN_ID}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_ROOT "${AUG_RUN_ROOT}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_ID "${RUN_ID}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" PREDICTIONS_DIR "${AUG_RUN_ROOT}/predictions"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" OUT_DIR "${AUG_RUN_ROOT}/tc_member_maps"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" EVENTS "${TC_EVENTS}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" DATE "${TC_MEMBER_MAPS_DATE}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" STEPS "${TC_MEMBER_MAPS_STEPS}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" MEMBERS "${TC_MEMBER_MAPS_MEMBERS}"
  set_var "${TC_MEMBER_MAPS_SCRIPT}" RUN_LABEL "${CHECKPOINT_SHORT}"
fi

bash -n "${INFER_SCRIPT}" "${SIGMA_SCRIPT}" "${WRITE_SCRIPT}"
if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
  bash -n "${TC_MEMBER_MAPS_SCRIPT}"
fi

SUBMIT_HOST_SHORT="$(hostname -s)"
case "${SUBMIT_HOST_SHORT}" in
  ac*) SUBMIT_HOST_FAMILY="ac" ;;
  ag*) SUBMIT_HOST_FAMILY="ag" ;;
  *) SUBMIT_HOST_FAMILY="unknown" ;;
esac

if [[ -n "${SUBMIT_QOS}" ]]; then
  EFFECTIVE_GPU_QOS="${SUBMIT_QOS_GPU:-${SUBMIT_QOS}}"
  EFFECTIVE_CPU_QOS="${SUBMIT_QOS_CPU:-${SUBMIT_QOS}}"
else
  if [[ -n "${SUBMIT_QOS_GPU}" ]]; then
    EFFECTIVE_GPU_QOS="${SUBMIT_QOS_GPU}"
  elif [[ "${SUBMIT_HOST_FAMILY}" == "ac" || "${SUBMIT_HOST_FAMILY}" == "ag" ]]; then
    EFFECTIVE_GPU_QOS="ng"
  else
    EFFECTIVE_GPU_QOS=""
  fi

  if [[ -n "${SUBMIT_QOS_CPU}" ]]; then
    EFFECTIVE_CPU_QOS="${SUBMIT_QOS_CPU}"
  elif [[ "${SUBMIT_HOST_FAMILY}" == "ac" ]]; then
    EFFECTIVE_CPU_QOS="nf"
  elif [[ "${SUBMIT_HOST_FAMILY}" == "ag" ]]; then
    EFFECTIVE_CPU_QOS="ng"
  else
    EFFECTIVE_CPU_QOS=""
  fi
fi

COMMON_SBATCH_ARGS=()
if [[ "${SUBMIT_HOLD}" -eq 1 ]]; then
  COMMON_SBATCH_ARGS+=(--hold)
fi
if [[ -n "${SUBMIT_TIME_OVERRIDE}" ]]; then
  COMMON_SBATCH_ARGS+=(--time "${SUBMIT_TIME_OVERRIDE}")
fi
if [[ -n "${SUBMIT_GPUS_OVERRIDE}" ]]; then
  COMMON_SBATCH_ARGS+=(--gpus-per-node "${SUBMIT_GPUS_OVERRIDE}")
fi

SIGMA_SBATCH_ARGS=("${COMMON_SBATCH_ARGS[@]}")
INFER_SBATCH_ARGS=("${COMMON_SBATCH_ARGS[@]}")
WRITE_SBATCH_ARGS=("${COMMON_SBATCH_ARGS[@]}")
if [[ -n "${EFFECTIVE_GPU_QOS}" ]]; then
  SIGMA_SBATCH_ARGS+=(--qos "${EFFECTIVE_GPU_QOS}")
  INFER_SBATCH_ARGS+=(--qos "${EFFECTIVE_GPU_QOS}")
fi
if [[ -n "${EFFECTIVE_CPU_QOS}" ]]; then
  WRITE_SBATCH_ARGS+=(--qos "${EFFECTIVE_CPU_QOS}")
fi

echo "[INFO] Submitting sigma..."
sigma_submit="$(sbatch "${SIGMA_SBATCH_ARGS[@]}" "${SIGMA_SCRIPT}")"
sigma_job="$(extract_job_id "${sigma_submit}")"
echo "[INFO] ${sigma_submit}"

echo "[INFO] Submitting inference..."
infer_submit="$(sbatch "${INFER_SBATCH_ARGS[@]}" "${INFER_SCRIPT}")"
infer_job="$(extract_job_id "${infer_submit}")"
echo "[INFO] ${infer_submit}"

echo "[INFO] Submitting post-inference writer (depends on sigma + inference)..."
write_submit="$(sbatch "${WRITE_SBATCH_ARGS[@]}" --dependency=afterok:${sigma_job}:${infer_job} "${WRITE_SCRIPT}")"
write_job="$(extract_job_id "${write_submit}")"
echo "[INFO] ${write_submit}"

tc_member_maps_job=""
if [[ -n "${TC_MEMBER_MAPS_SCRIPT}" ]]; then
  echo "[INFO] Submitting TC member maps (depends on inference)..."
  tc_mm_submit="$(sbatch "${WRITE_SBATCH_ARGS[@]}" --dependency=afterok:${infer_job} "${TC_MEMBER_MAPS_SCRIPT}")"
  tc_member_maps_job="$(extract_job_id "${tc_mm_submit}")"
  echo "[INFO] ${tc_mm_submit}"
fi

# --- Finalize: lean eval layout reorganization ---
FINALIZE_TEMPLATE="${TEMPLATE_DIR}/finalize_lean_eval_layout.sbatch"
FINALIZE_SCRIPT="${SUBMIT_DIR}/finalize_lean_eval_${RUN_ID}.sbatch"
AUG_RUN_ROOT="/home/ecm5702/scratch/eval/${RUN_ID}"
finalize_job="skipped"
if [[ -f "${FINALIZE_TEMPLATE}" ]]; then
  if [[ ! -f "${FINALIZE_SCRIPT}" ]] || [[ "${ALLOW_OVERWRITE}" -eq 1 ]]; then
    cp "${FINALIZE_TEMPLATE}" "${FINALIZE_SCRIPT}"
    set_var "${FINALIZE_SCRIPT}" RUN_ROOT "${AUG_RUN_ROOT}"
    set_var "${FINALIZE_SCRIPT}" RUN_ID "${RUN_ID}"
  fi
  finalize_dep="${write_job}"
  if [[ -n "${tc_member_maps_job}" ]]; then
    finalize_dep="${finalize_dep}:${tc_member_maps_job}"
  fi
  finalize_submit="$(sbatch "${WRITE_SBATCH_ARGS[@]}" --dependency=afterok:${finalize_dep} "${FINALIZE_SCRIPT}")"
  finalize_job="$(extract_job_id "${finalize_submit}")"
  echo "[INFO] ${finalize_submit}"
fi

cat <<EOF

=== AUG 26-30 SCOREBOARD FLOW SUBMITTED ===
run_id:            ${RUN_ID}
checkpoint_ref:    ${CHECKPOINT_REF}
submit_qos:        ${SUBMIT_QOS:-auto}
submit_qos_gpu:    ${EFFECTIVE_GPU_QOS:-template-default}
submit_qos_cpu:    ${EFFECTIVE_CPU_QOS:-template-default}
submit_hold:       ${SUBMIT_HOLD}
submit_time:       ${SUBMIT_TIME_OVERRIDE:-template-default}
submit_gpus:       ${SUBMIT_GPUS_OVERRIDE:-template-default}
generated_scripts:
  - ${INFER_SCRIPT}
  - ${SIGMA_SCRIPT}
  - ${WRITE_SCRIPT}
job_ids:
  sigma:           ${sigma_job}
  inference:       ${infer_job}
  post_writer:     ${write_job} (afterok:${sigma_job}:${infer_job})
  tc_member_maps:   ${tc_member_maps_job:-skipped} (afterok:${infer_job})
  finalize_lean:   ${finalize_job}

Monitor:
  squeue -j ${sigma_job},${infer_job},${write_job}${tc_member_maps_job:+,${tc_member_maps_job}}${finalize_job:+,${finalize_job}}
EOF
