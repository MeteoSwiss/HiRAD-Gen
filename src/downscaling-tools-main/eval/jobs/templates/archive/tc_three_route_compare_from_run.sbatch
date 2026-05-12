#!/bin/bash --login
# Canonical representative-TC comparison template for O320->O1280 runs.
#
# Route 1: storm-centered six-panel local plots from predictions_*.nc
# Route 2: contour-style TC six-panel plots from predictions_*.nc
# Route 3: center-track tcstyle/regionstyle plots from a selected intermediate bundle
#
# This template selects one representative case per requested event from the
# finished predictions tree, then renders all three routes into a shared output
# directory so the storm-style preference can be judged on like-for-like cases.
#
# Canonical host/env rule for new-stack O1280 work:
#   - AG login host -> `/home/ecm5702/dev/.ds-ag/bin/activate`
#   - AC login host -> `/home/ecm5702/dev/.ds-dyn/bin/activate`
# Never reuse the env choice from the wrong host family.

#SBATCH --job-name=o1280_tc3cmp
#SBATCH --qos=ng
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --mem=256G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
PREDICTIONS_DIR="${RUN_ROOT}/predictions"
BUNDLE_DIR="${RUN_ROOT}/bundles_with_y"
CHECKPOINT_REF="REPLACE_CHECKPOINT_PATH"
OUT_DIR="${RUN_ROOT}/tc_three_route_compare"
EVENTS="idalia,franklin"
SELECTION_METRIC="maxwind"        # maxwind | minmsl
SELECTION_REGION=""               # blank => selector default; otherwise pass explicit region name
PREDICTIONS_FILE_OVERRIDE=""      # optional explicit predictions_*.nc for single-event reruns
BUNDLE_FILE_OVERRIDE=""           # optional explicit bundle for route-3 reruns
ENSEMBLE_MEMBER_INDEX_OVERRIDE="" # optional 0-based member index to pair with override files
CASE_TAG_OVERRIDE=""              # optional stable tag when override files are used
SAMPLE_INDEX=0
MODEL_VARIABLES="x_0,x_interp_0,y_0,y_pred_0,residuals_0,residuals_pred_0"
INTERMEDIATE_STEPS="-1,0,10,20,30,39"
INTERMEDIATE_CAPTURE_STEPS="0,5,10,15,20,25,30,35,39"
INTERMEDIATE_WEATHER_STATES="msl,10u"
INTERMEDIATE_PRECISION="fp16"     # fp32 | fp16 | bf16
INTERMEDIATE_DEVICE="cuda"
ALSO_PNG=1
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"
SELECT_CASE_SCRIPT="${PROJECT_ROOT}/eval/jobs/select_tc_representative_case.py"
GENERATE_INTERMEDIATE_SCRIPT="${PROJECT_ROOT}/eval/jobs/generate_intermediate_from_bundle.py"


die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*)
    CLUSTER="ac"
    source /home/ecm5702/dev/.ds-dyn/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-dyn/bin/python
    ;;
  ag*)
    CLUSTER="ag"
    source /home/ecm5702/dev/.ds-ag/bin/activate
    PYTHON=/home/ecm5702/dev/.ds-ag/bin/python
    ;;
  *) die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

[[ -x "${PYTHON}" ]] || die "Expected host-matched Python not found: ${PYTHON}"
[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ "${CHECKPOINT_REF}" != REPLACE_* ]] || die "Set CHECKPOINT_REF."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ -d "${PREDICTIONS_DIR}" ]] || die "PREDICTIONS_DIR does not exist: ${PREDICTIONS_DIR}"
[[ -d "${BUNDLE_DIR}" ]] || die "BUNDLE_DIR does not exist: ${BUNDLE_DIR}"
[[ -f "${SELECT_CASE_SCRIPT}" ]] || die "Missing selector script: ${SELECT_CASE_SCRIPT}"
[[ -f "${GENERATE_INTERMEDIATE_SCRIPT}" ]] || die "Missing intermediate generator: ${GENERATE_INTERMEDIATE_SCRIPT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

mkdir -p "${OUT_DIR}" "${RUN_ROOT}/logs"

IFS=',' read -r -a EVENT_LIST <<< "${EVENTS}"
IFS=',' read -r -a INTERMEDIATE_STATE_LIST <<< "${INTERMEDIATE_WEATHER_STATES}"

MANIFEST_PATH="${OUT_DIR}/manifest.jsonl"
: > "${MANIFEST_PATH}"

echo "[INFO] host=${HOST_SHORT} cluster=${CLUSTER}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] checkpoint_ref=${CHECKPOINT_REF}"
echo "[INFO] events=${EVENTS}"
echo "[INFO] selection_metric=${SELECTION_METRIC}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

for raw_event in "${EVENT_LIST[@]}"; do
  event="$(printf '%s' "${raw_event}" | xargs)"
  [[ -n "${event}" ]] || continue
  event_dir="${OUT_DIR}/${event}"
  storm_dir="${event_dir}/storm_six_panel"
  contour_dir="${event_dir}/contour_six_panel"
  centertrack_dir="${event_dir}/centertrack_intermediate"
  mkdir -p "${event_dir}" "${storm_dir}" "${contour_dir}" "${centertrack_dir}"

  select_json="${event_dir}/selected_case.json"
  if [[ -n "${PREDICTIONS_FILE_OVERRIDE}" || -n "${BUNDLE_FILE_OVERRIDE}" || -n "${ENSEMBLE_MEMBER_INDEX_OVERRIDE}" ]]; then
    [[ -n "${PREDICTIONS_FILE_OVERRIDE}" ]] || die "Set PREDICTIONS_FILE_OVERRIDE when using explicit overrides."
    [[ -n "${BUNDLE_FILE_OVERRIDE}" ]] || die "Set BUNDLE_FILE_OVERRIDE when using explicit overrides."
    [[ -n "${ENSEMBLE_MEMBER_INDEX_OVERRIDE}" ]] || die "Set ENSEMBLE_MEMBER_INDEX_OVERRIDE when using explicit overrides."
    predictions_file="${PREDICTIONS_FILE_OVERRIDE}"
    bundle_file="${BUNDLE_FILE_OVERRIDE}"
    member_index="${ENSEMBLE_MEMBER_INDEX_OVERRIDE}"
    case_tag="${CASE_TAG_OVERRIDE:-${event}_explicit_case}"
    cat > "${select_json}" <<JSON
{
  "event": "${event}",
  "metric": "${SELECTION_METRIC}",
  "region": "${SELECTION_REGION:-${event}}",
  "predictions_file": "${predictions_file}",
  "bundle_file": "${bundle_file}",
  "ensemble_member_index": ${member_index},
  "case_tag": "${case_tag}",
  "selection_mode": "explicit_override"
}
JSON
  else
    select_cmd=(
      "${PYTHON}" "${SELECT_CASE_SCRIPT}"
      --predictions-dir "${PREDICTIONS_DIR}"
      --bundle-dir "${BUNDLE_DIR}"
      --event "${event}"
      --metric "${SELECTION_METRIC}"
      --sample-index "${SAMPLE_INDEX}"
      --out-json "${select_json}"
    )
    if [[ -n "${SELECTION_REGION}" ]]; then
      select_cmd+=(--region "${SELECTION_REGION}")
    fi
    "${select_cmd[@]}"
    predictions_file="$(jq -r '.predictions_file' "${select_json}")"
    bundle_file="$(jq -r '.bundle_file' "${select_json}")"
    member_index="$(jq -r '.ensemble_member_index' "${select_json}")"
    case_tag="$(jq -r '.case_tag' "${select_json}")"
  fi

  [[ -f "${predictions_file}" ]] || die "Selected predictions file missing: ${predictions_file}"
  [[ -f "${bundle_file}" ]] || die "Selected bundle file missing: ${bundle_file}"

  storm_cmd=(
    "${PYTHON}" -m eval._backends.region_plotting.plot_regions
    --predictions-nc "${predictions_file}"
    --out-dir "${storm_dir}"
    --regions "${event}"
    --sample-index "${SAMPLE_INDEX}"
    --ensemble-member-index "${member_index}"
    --model-variables "${MODEL_VARIABLES}"
    --suite-kind storm
  )
  contour_cmd=(
    "${PYTHON}" -m eval._backends.region_plotting.plot_tc_contours_from_predictions
    --predictions-nc "${predictions_file}"
    --out-dir "${contour_dir}"
    --regions "${event}"
    --sample-index "${SAMPLE_INDEX}"
    --ensemble-member-index "${member_index}"
  )
  if [[ "${ALSO_PNG}" == "1" ]]; then
    storm_cmd+=(--also-png)
    contour_cmd+=(--also-png)
  fi

  echo "[INFO] event=${event} route=storm file=${predictions_file} member_index=${member_index}"
  "${storm_cmd[@]}"
  echo "[INFO] event=${event} route=contour file=${predictions_file} member_index=${member_index}"
  "${contour_cmd[@]}"

  intermediate_nc="${centertrack_dir}/${case_tag}__intermediate_cached.nc"
  echo "[INFO] event=${event} route=centertrack bundle=${bundle_file}"
  srun --ntasks=4 --ntasks-per-node=4 "${PYTHON}" "${GENERATE_INTERMEDIATE_SCRIPT}"     --bundle-nc "${bundle_file}"     --out-nc "${intermediate_nc}"     --ckpt-ref "${CHECKPOINT_REF}"     --device "${INTERMEDIATE_DEVICE}"     --num-gpus-per-model 4     --precision "${INTERMEDIATE_PRECISION}"     --capture-steps "${INTERMEDIATE_CAPTURE_STEPS}"     --include-init-state

  for weather_state in "${INTERMEDIATE_STATE_LIST[@]}"; do
    ws="$(printf '%s' "${weather_state}" | xargs)"
    [[ -n "${ws}" ]] || continue
    tcstyle_out="${centertrack_dir}/${ws}_${event}_centertrack_tcstyle.png"
    regionstyle_out="${centertrack_dir}/${ws}_${event}_centertrack_regionstyle.png"
    stats_out="${centertrack_dir}/${ws}_${event}_centertrack_stats.json"
    cmd=(
      "${PYTHON}" -m eval._backends.plot_intermediate.plot_intermediate dataset
      --predictions-nc "${intermediate_nc}"
      --sample 0
      --weather-state "${ws}"
      --region "${event}"
      --steps "${INTERMEDIATE_STEPS}"
      --max-cols 3
      --center-track-mode min
      --center-track-state msl
      --center-window-deg 6
      --annotate-extremes
      --extreme-radius-deg 3
      --tc-member-style
      --independent-color-scales
      --panel-scale-mode minmax
      --stats-out "${stats_out}"
      --also-region-style-out "${regionstyle_out}"
      --out "${tcstyle_out}"
    )
    echo "[INFO] event=${event} route=centertrack weather_state=${ws}"
    "${cmd[@]}"
  done

  jq -c '.' "${select_json}" >> "${MANIFEST_PATH}"
done

echo "DONE: ${OUT_DIR}"
