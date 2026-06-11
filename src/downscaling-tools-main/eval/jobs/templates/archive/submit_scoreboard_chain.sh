#!/bin/bash
# DEPRECATED: Use python -m eval.jobs.pipeline instead.
# This template is kept for historical reference. New evaluation runs
# should use the unified CLI (python -m eval.cli) and pipeline renderer.
# Launcher for the four scoreboard step sbatches.

set -euo pipefail

RUN_ID="${RUN_ID:-REPLACE_RUN_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ecm5702/scratch/eval}"
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_ROOT="${SUBMIT_ROOT:-/home/ecm5702/dev/jobscripts/submit/$(date -u +%Y%m%d)}"

[[ "${RUN_ID}" != REPLACE_* ]] || { echo "ERROR: set RUN_ID" >&2; exit 2; }
source "${TEMPLATE_DIR}/render_helpers.sh"

mkdir -p "${SUBMIT_ROOT}"
SPECTRA_SCRIPT="${SUBMIT_ROOT}/spectra_${RUN_ID}.sbatch"
TC_SCRIPT="${SUBMIT_ROOT}/tc_${RUN_ID}.sbatch"
SFC_SCRIPT="${SUBMIT_ROOT}/surface_${RUN_ID}.sbatch"
FIN_SCRIPT="${SUBMIT_ROOT}/finalize_${RUN_ID}.sbatch"

cp "${TEMPLATE_DIR}/scoreboard_spectra_step.sbatch" "${SPECTRA_SCRIPT}"
cp "${TEMPLATE_DIR}/scoreboard_tc_step.sbatch" "${TC_SCRIPT}"
cp "${TEMPLATE_DIR}/scoreboard_surface_step.sbatch" "${SFC_SCRIPT}"
cp "${TEMPLATE_DIR}/scoreboard_finalize_step.sbatch" "${FIN_SCRIPT}"

for script in "${SPECTRA_SCRIPT}" "${TC_SCRIPT}" "${SFC_SCRIPT}" "${FIN_SCRIPT}"; do
  set_var "${script}" RUN_ID "${RUN_ID}"
  set_var "${script}" OUTPUT_ROOT "${OUTPUT_ROOT}"
done

spectra_sub="$(sbatch --parsable "${SPECTRA_SCRIPT}")"
tc_sub="$(sbatch --parsable "${TC_SCRIPT}")"
sfc_sub="$(sbatch --parsable "${SFC_SCRIPT}")"
fin_sub="$(sbatch --parsable --dependency=afterok:${spectra_sub}:${tc_sub}:${sfc_sub} "${FIN_SCRIPT}")"

cat <<EOF
=== SCOREBOARD CHAIN SUBMITTED ===
run_id:    ${RUN_ID}
spectra:   ${spectra_sub}
tc:        ${tc_sub}
surface:   ${sfc_sub}
finalize:  ${fin_sub} (afterok:${spectra_sub}:${tc_sub}:${sfc_sub})

Monitor:
  squeue -j ${spectra_sub},${tc_sub},${sfc_sub},${fin_sub}
EOF
