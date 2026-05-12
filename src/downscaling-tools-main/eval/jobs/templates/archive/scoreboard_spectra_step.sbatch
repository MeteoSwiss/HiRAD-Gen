#!/bin/bash --login
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
#SBATCH --job-name=scoreboard_spectra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

RUN_ID="REPLACE_RUN_ID"
LANE="o96_o320"
OUTPUT_ROOT="/home/ecm5702/scratch/eval"
RUN_ROOT=""
PRED_DIR=""
EXPECTED_PRED_FILES=25
SPECTRA_METHOD="proxy"
SPECTRA_BUNDLE_PAIRS="20230829:24,20230828:48,20230829:48,20230828:24,20230830:24,20230828:72,20230827:72,20230830:48,20230829:72,20230827:48"
SPECTRA_WEATHER_STATES_PROXY="10u,10v,2t,msl,t_850,z_500"
SPECTRA_NSIDE=128
SPECTRA_LMAX=319
SPECTRA_MEMBER_AGG="raw-members"

[[ "${RUN_ID}" != REPLACE_* ]] || { echo "ERROR: set RUN_ID" >&2; exit 2; }
[[ -n "${RUN_ROOT}" ]] || RUN_ROOT="${OUTPUT_ROOT}/${RUN_ID}"
[[ -n "${PRED_DIR}" ]] || PRED_DIR="${RUN_ROOT}/predictions"
REPO_ROOT="${DOWNSCALING_TOOLS_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
SPECTRA_OUT="${RUN_ROOT}/data/spectra"
SUBSET_DIR="${RUN_ROOT}/predictions_proxy10_subset"
mkdir -p "${SPECTRA_OUT}" "${SUBSET_DIR}"

IFS="," read -r -a pairs <<< "${SPECTRA_BUNDLE_PAIRS}"
for pair in "${pairs[@]}"; do
  date="${pair%%:*}"
  step="${pair##*:}"
  printf -v step_padded "%03d" "${step}"
  src="${PRED_DIR}/predictions_${date}_step${step_padded}.nc"
  [[ -f "${src}" ]] || { echo "ERROR: missing spectra input ${src}" >&2; exit 2; }
  ln -sfn "${src}" "${SUBSET_DIR}/predictions_${date}_step${step_padded}.nc"
done

source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python "${REPO_ROOT}/eval/jobs/templates/predictions_dir_spectra.py" \
  --predictions-dir "${SUBSET_DIR}" \
  --out-dir "${SPECTRA_OUT}" \
  --run-label "${RUN_ID}_proxy10_subset" \
  --weather-states "${SPECTRA_WEATHER_STATES_PROXY}" \
  --nside "${SPECTRA_NSIDE}" \
  --lmax "${SPECTRA_LMAX}" \
  --member-aggregation "${SPECTRA_MEMBER_AGG}"

echo "DONE scoreboard_spectra_step run_id=${RUN_ID}"
echo "SPECTRA_SUMMARY=${SPECTRA_OUT}/spectra_summary.json"
