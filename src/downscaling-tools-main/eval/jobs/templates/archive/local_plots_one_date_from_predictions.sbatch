#!/bin/bash --login
# Canonical one-date local-plot template — works on both AC and AG.
#
# Renders canonical non-TC local plots from an existing predictions tree using:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/region_plotting/plot_one_date_local.py
#
# Supports two render modes:
#   RENDER_MODE=single  — renders a single region per prediction file via
#                         plot_one_date_local (default, backward compatible).
#   RENDER_MODE=suite   — renders a list of regions via render_region_suite_from_predictions_file
#                         from a single prediction file (first step). Canonical mode for o1280→o2560.
#
# Preferred launch path for the `o320 -> o1280` manual-eval lane:
#   /etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools/eval/jobs/templates/submit_o320_o1280_manual_eval_flow.sh
#
# Raw submission compatibility note:
#   the directives below stay cross-cluster-compatible for deliberate direct
#   `sbatch`, but the preferred helper path patches them to a host-safe CPU
#   posture (`nf` without GPUs on AC, `ng` with `--gpus-per-node=0` on AG).
#
# Canonical host/env rule for new-stack O1280 work:
#   - AG login host -> `/home/ecm5702/dev/.ds-ag/bin/activate`
#   - AC login host -> `/home/ecm5702/dev/.ds-dyn/bin/activate`
# Never reuse the env choice from the wrong host family.

#SBATCH --job-name=local_plots_one_date
#SBATCH --qos=ng
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="/home/ecm5702/scratch/eval/REPLACE_RUN_ID"
RUN_ID="REPLACE_RUN_ID"
DATE="20230826"
OUT_SUBDIR="local_plots_one_date"
REGION="auto"
SAMPLE_INDEX=0
ENSEMBLE_MEMBER_INDEX=0
EXPECTED_COUNT=5
MODEL_VARIABLES=""          # blank => plot_one_date_local defaults
WEATHER_STATES=""           # blank => plot_one_date_local defaults
# Render mode: "single" (one region per file, all steps) or "suite" (multi-region, first step).
RENDER_MODE="single"
# Space-separated region names for suite mode. Blank => use grid default regions.
REGION_LIST=""
###############################################################################

PROJECT_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) CLUSTER="ac" ;;
  ag*) CLUSTER="ag" ;;
  *) die "Unknown cluster from hostname '${HOST_SHORT}'." ;;
esac

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"

PREDICTIONS_DIR="${RUN_ROOT}/predictions"
[[ -d "${PREDICTIONS_DIR}" ]] || die "Predictions directory does not exist: ${PREDICTIONS_DIR}"

mkdir -p "${RUN_ROOT}/logs"

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
[[ -x "${PYTHON}" ]] || die "Expected host-matched Python not found: ${PYTHON}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[INFO] cluster=${CLUSTER} host=${HOST_SHORT}"
echo "[INFO] run_root=${RUN_ROOT}"
echo "[INFO] date=${DATE}"
echo "[INFO] render_mode=${RENDER_MODE}"
echo "[INFO] expected_count=${EXPECTED_COUNT}"
echo "[INFO] VIRTUAL_ENV=${VIRTUAL_ENV:-unset}"
echo "[INFO] python=$(which python) ($("${PYTHON}" -V 2>&1))"

if [[ "${RENDER_MODE}" == "suite" ]]; then
  # Suite mode: render multiple regions from the first prediction step file.
  # Each region is rendered sequentially with gc between to stay within memory.
  PRED_FILE="$(ls -1 "${PREDICTIONS_DIR}/predictions_${DATE}_step"*.nc 2>/dev/null | head -1)"
  [[ -n "${PRED_FILE}" ]] || die "No prediction file found for date=${DATE} in ${PREDICTIONS_DIR}"
  echo "[INFO] suite mode: pred_file=${PRED_FILE}"
  echo "[INFO] region_list=${REGION_LIST}"

  WS_ARG=""
  [[ -n "${WEATHER_STATES}" ]] && WS_ARG="weather_states='${WEATHER_STATES}'.split(',')"

  "${PYTHON}" -c "
import gc, sys
sys.path.insert(0, '${PROJECT_ROOT}')
from eval._backends.region_plotting.plot_regions import render_region_suite_from_predictions_file
region_list = '${REGION_LIST}'.split() if '${REGION_LIST}'.strip() else None
ws = [s.strip() for s in '${WEATHER_STATES}'.split(',')] if '${WEATHER_STATES}'.strip() else None
if region_list:
    for i, r in enumerate(region_list):
        print(f'[{i+1}/{len(region_list)}] {r}')
        render_region_suite_from_predictions_file(
            predictions_nc='${PRED_FILE}',
            out_dir='${RUN_ROOT}/${OUT_SUBDIR}',
            region_names=[r],
            weather_states=ws,
        )
        gc.collect()
else:
    render_region_suite_from_predictions_file(
        predictions_nc='${PRED_FILE}',
        out_dir='${RUN_ROOT}/${OUT_SUBDIR}',
        weather_states=ws,
    )
# Merge individual region PDFs into all_regions_plots.pdf
import os
from pypdf import PdfWriter
writer = PdfWriter()
out_dir = '${RUN_ROOT}/${OUT_SUBDIR}'
merge_list = region_list if region_list else [
    os.path.splitext(f)[0] for f in sorted(os.listdir(out_dir))
    if f.endswith('.pdf') and f != 'all_regions_plots.pdf'
]
for r in merge_list:
    pdf_path = os.path.join(out_dir, f'{r}.pdf')
    if os.path.exists(pdf_path):
        writer.append(pdf_path)
combined = os.path.join(out_dir, 'all_regions_plots.pdf')
writer.write(combined)
writer.close()
print(f'Merged {len(merge_list)} regions into {combined}')
print('Suite rendering complete.')
"
else
  # Single mode: render one region per prediction file for all steps.
  CMD=(
    "${PYTHON}" -m eval._backends.region_plotting.plot_one_date_local
    --run-root "${RUN_ROOT}"
    --date "${DATE}"
    --out-subdir "${OUT_SUBDIR}"
    --region "${REGION}"
    --sample-index "${SAMPLE_INDEX}"
    --ensemble-member-index "${ENSEMBLE_MEMBER_INDEX}"
    --expected-count "${EXPECTED_COUNT}"
    --run-label "${RUN_ID}"
  )

  if [[ -n "${MODEL_VARIABLES}" ]]; then
    CMD+=(--model-variables "${MODEL_VARIABLES}")
  fi
  if [[ -n "${WEATHER_STATES}" ]]; then
    CMD+=(--weather-states "${WEATHER_STATES}")
  fi

  echo "[INFO] Running: ${CMD[*]}"
  "${CMD[@]}"
fi

OUT_ROOT="${RUN_ROOT}/${OUT_SUBDIR}"
[[ -d "${OUT_ROOT}" ]] || die "Expected output directory missing: ${OUT_ROOT}"

PDF_COUNT="$(find "${OUT_ROOT}" -type f -name '*.pdf' | wc -l | tr -d ' ')"
PNG_COUNT="$(find "${OUT_ROOT}" -type f -name '*.png' | wc -l | tr -d ' ')"
[[ "${PDF_COUNT}" -gt 0 ]] || die "No PDF outputs found under ${OUT_ROOT}"
[[ "${PNG_COUNT}" -gt 0 ]] || die "No PNG outputs found under ${OUT_ROOT}"

echo "DONE: ${OUT_ROOT} (pdf=${PDF_COUNT} png=${PNG_COUNT})"
