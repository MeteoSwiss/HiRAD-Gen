#!/bin/bash --login
# Finalize lean eval layout — reorganize a run root into the lean PDF bundle format.
#
# This script is meant to run AFTER all eval jobs (predictions, spectra, TC, local plots)
# have completed. It:
#   1. Creates a data/ subdirectory and moves raw artifacts into it
#   2. Copies/links consolidated PDFs to the run-root top level with clean names
#   3. Keeps EXPERIMENT_CONFIG.yaml and training diagnostics at the top level
#
# Canonical submission:
#   Submit with afterok dependency on all other eval jobs:
#     sbatch --dependency=afterok:<JOB1>:<JOB2>:... finalize_lean_eval_layout.sbatch
#
# Can also be run directly on a login node for existing run roots:
#   bash finalize_lean_eval_layout.sbatch

#SBATCH --job-name=finalize_lean
#SBATCH --qos=nf
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=4G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
RUN_ROOT="${RUN_ROOT:-/home/ecm5702/scratch/eval/REPLACE_RUN_ID}"
RUN_ID="${RUN_ID:-REPLACE_RUN_ID}"
LANE="${LANE:-}"                   # e.g. o48_o96, o96_o320, o320_o1280, o1280_o2560

# Spectra consolidated PDF names at top level
SPECTRA_PROXY_PDF_NAME="spectra_proxy.pdf"
SPECTRA_ECMWF_PDF_NAME="spectra_ecmwf.pdf"

# Local plot PDF names at top level (one per step)
LOCAL_PLOT_STEPS="24,120"

# When enabled, recreate the old top-level raw paths as symlinks into data/.
# O48->O96 currently disables this so the run root stays visually lean.
CREATE_BACKCOMPAT_SYMLINKS="${CREATE_BACKCOMPAT_SYMLINKS:-1}"

# TC consolidated PDF names
TC_CONTOUR_PDF_NAME="tc_contours.pdf"
TC_PDF_DIST_NAME="tc_pdf_distributions.pdf"
# TC member maps: copied per-event (tc_member_maps_<event>.pdf) — no merge needed
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

maybe_link_backcompat() {
  local target="$1"
  local link_path="$2"
  if [[ "${CREATE_BACKCOMPAT_SYMLINKS}" == "1" ]]; then
    ln -sf "${target}" "${link_path}" 2>/dev/null || true
  fi
}

[[ "${RUN_ROOT}" != *REPLACE_* ]] || die "Set RUN_ROOT."
[[ "${RUN_ID}" != REPLACE_* ]] || die "Set RUN_ID."
[[ -d "${RUN_ROOT}" ]] || die "RUN_ROOT does not exist: ${RUN_ROOT}"
[[ "${CREATE_BACKCOMPAT_SYMLINKS}" == "0" || "${CREATE_BACKCOMPAT_SYMLINKS}" == "1" ]] || die "CREATE_BACKCOMPAT_SYMLINKS must be 0 or 1."

DATA_DIR="${RUN_ROOT}/data"
mkdir -p "${DATA_DIR}"

echo "[INFO] Reorganizing ${RUN_ROOT} into lean PDF bundle layout"
echo "[INFO] data/ directory: ${DATA_DIR}"

###############################################################################
# STEP 1: Move raw artifact directories into data/
###############################################################################
DIRS_TO_MOVE=(
  predictions
  predictions_manifest.csv
  bundles_with_y
  logs
  manual_inference_run_info.txt
  bundle_build_verification.json
  bundle_preflight.json
  checkpoint_profile.json
  sigma_eval_table.csv
  training_loss_plots_status.json
)
# NOTE: scoreboard_metrics.json and surface_loss_summary.json are now symlinks
# to canonical locations and should NOT be moved. Skip them if they are symlinks.
for legacy_link in "scoreboard_metrics.json" "surface_loss_summary.json"; do
  src="${RUN_ROOT}/${legacy_link}"
  if [[ -e "${src}" ]] && [[ ! -L "${src}" ]] && [[ ! -e "${DATA_DIR}/${legacy_link}" ]]; then
    mv "${src}" "${DATA_DIR}/${legacy_link}"
    maybe_link_backcompat "data/${legacy_link}" "${src}"
    echo "[MOVE] ${legacy_link} -> data/${legacy_link}"
  fi
done

for item in "${DIRS_TO_MOVE[@]}"; do
  src="${RUN_ROOT}/${item}"
  dst="${DATA_DIR}/${item}"
  if [[ -e "${src}" ]] && [[ ! -e "${dst}" ]]; then
    mv "${src}" "${dst}"
    maybe_link_backcompat "data/${item}" "${src}"
    echo "[MOVE] ${item} -> data/${item}"
  fi
done

# Move lane-specific or run-specific metrics JSON files that are not covered by
# the historical scoreboard_metrics.json name.
for metrics_file in "${RUN_ROOT}"/*_metrics.json; do
  [[ -f "${metrics_file}" ]] || continue
  base="$(basename "${metrics_file}")"
  dst="${DATA_DIR}/${base}"
  if [[ ! -e "${dst}" ]]; then
    mv "${metrics_file}" "${dst}"
    maybe_link_backcompat "data/${base}" "${RUN_ROOT}/${base}"
    echo "[MOVE] ${base} -> data/${base}"
  fi
done

# Move spectra raw data (skip symlinks — canonical layout uses symlinks for compat)
for spectra_dir in "${RUN_ROOT}"/spectra_*/ "${RUN_ROOT}"/spectra_*/; do
  [[ -d "${spectra_dir}" ]] || continue
  [[ ! -L "${spectra_dir%/}" ]] || continue
  dir_name="$(basename "${spectra_dir}")"
  # Skip if already under data/
  [[ "${spectra_dir}" != "${DATA_DIR}/"* ]] || continue
  dst="${DATA_DIR}/${dir_name}"
  if [[ ! -e "${dst}" ]]; then
    mv "${spectra_dir}" "${dst}"
    maybe_link_backcompat "data/${dir_name}" "${RUN_ROOT}/${dir_name}"
    echo "[MOVE] ${dir_name} -> data/${dir_name}"
  fi
done

# Move local plot raw directories
for plot_dir in "${RUN_ROOT}"/local_plots_*/ "${RUN_ROOT}"/storm_local_plots_*/ "${RUN_ROOT}"/tc_local_plots_*/ "${RUN_ROOT}"/tc_member_maps*/; do
  [[ -d "${plot_dir}" ]] || continue
  dir_name="$(basename "${plot_dir}")"
  [[ "${plot_dir}" != "${DATA_DIR}/"* ]] || continue
  dst="${DATA_DIR}/${dir_name}"
  if [[ ! -e "${dst}" ]]; then
    mv "${plot_dir}" "${dst}"
    maybe_link_backcompat "data/${dir_name}" "${RUN_ROOT}/${dir_name}"
    echo "[MOVE] ${dir_name} -> data/${dir_name}"
  fi
done

# Move prediction subset directories
for subset_dir in "${RUN_ROOT}"/predictions_*/ "${RUN_ROOT}"/predictions_subset_*/; do
  [[ -d "${subset_dir}" ]] || continue
  dir_name="$(basename "${subset_dir}")"
  [[ "${subset_dir}" != "${DATA_DIR}/"* ]] || continue
  # Skip if it's the top-level predictions dir (already handled)
  [[ "${dir_name}" != "predictions" ]] || continue
  dst="${DATA_DIR}/${dir_name}"
  if [[ ! -e "${dst}" ]]; then
    mv "${subset_dir}" "${dst}"
    maybe_link_backcompat "data/${dir_name}" "${RUN_ROOT}/${dir_name}"
    echo "[MOVE] ${dir_name} -> data/${dir_name}"
  fi
done

# Move baseline reference directories (enfo_o320, eefo_o96, etc.)
for ref_dir in "${RUN_ROOT}"/enfo_*/ "${RUN_ROOT}"/eefo_*/; do
  [[ -d "${ref_dir}" ]] || continue
  dir_name="$(basename "${ref_dir}")"
  [[ "${ref_dir}" != "${DATA_DIR}/"* ]] || continue
  dst="${DATA_DIR}/${dir_name}"
  if [[ ! -e "${dst}" ]]; then
    mv "${ref_dir}" "${dst}"
    maybe_link_backcompat "data/${dir_name}" "${RUN_ROOT}/${dir_name}"
    echo "[MOVE] ${dir_name} -> data/${dir_name}"
  fi
done

# Move TC stats JSON files and raw TC PDFs into data/ (skip symlinks from canonical layout)
for tc_file in "${RUN_ROOT}"/tc_normed_pdfs_*.stats.json "${RUN_ROOT}"/tc_normed_pdfs_*.pdf; do
  [[ -f "${tc_file}" ]] || continue
  [[ ! -L "${tc_file}" ]] || continue
  base="$(basename "${tc_file}")"
  dst="${DATA_DIR}/${base}"
  if [[ ! -e "${dst}" ]]; then
    mv "${tc_file}" "${dst}"
    maybe_link_backcompat "data/${base}" "${RUN_ROOT}/${base}"
    echo "[MOVE] ${base} -> data/${base}"
  fi
done

###############################################################################
# STEP 2: Copy consolidated PDFs to top level with clean names
###############################################################################

# --- Local plots: one PDF per step ---
IFS=',' read -ra STEP_LIST <<< "${LOCAL_PLOT_STEPS}"
for step in "${STEP_LIST[@]}"; do
  step="$(echo "${step}" | tr -d ' ')"
  step_padded="$(printf '%03d' "${step}")"
  # Look for the all_regions_plots.pdf in the regional suite dir for this step
  for candidate_dir in \
    "${DATA_DIR}/local_plots_regions_step${step_padded}" \
    "${DATA_DIR}/local_plots_regions_step${step}" \
    "${RUN_ROOT}/local_plots_regions_step${step_padded}" \
    "${RUN_ROOT}/local_plots_regions_step${step}"; do
    combined="${candidate_dir}/all_regions_plots.pdf"
    if [[ -f "${combined}" ]]; then
      cp "${combined}" "${RUN_ROOT}/local_plots_step${step_padded}.pdf"
      echo "[PDF] local_plots_step${step_padded}.pdf (from ${combined})"
      break
    fi
  done
done

# --- Spectra proxy consolidated PDF ---
# Check if --consolidated-pdf was already used (file might already exist at top level)
if [[ ! -f "${RUN_ROOT}/${SPECTRA_PROXY_PDF_NAME}" ]]; then
  # Try to merge per-variable proxy spectra PDFs
  PROXY_PDFS=()
  for candidate_dir in \
    "${DATA_DIR}/spectra_proxy"* \
    "${DATA_DIR}/spectra_o48_o96" \
    "${RUN_ROOT}/spectra_proxy"* \
    "${RUN_ROOT}/spectra_o48_o96"; do
    [[ -d "${candidate_dir}" ]] || continue
    for pdf_file in "${candidate_dir}"/spectra_*.pdf; do
      [[ -f "${pdf_file}" ]] || continue
      PROXY_PDFS+=("${pdf_file}")
    done
    if [[ ${#PROXY_PDFS[@]} -gt 0 ]]; then
      break
    fi
  done

  if [[ ${#PROXY_PDFS[@]} -gt 0 ]]; then
    # Try pypdf merge via Python
    python3 -c "
import sys
try:
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for src in sys.argv[1:]:
        for page in PdfReader(src).pages:
            writer.add_page(page)
    with open('${RUN_ROOT}/${SPECTRA_PROXY_PDF_NAME}', 'wb') as f:
        writer.write(f)
    print('[PDF] ${SPECTRA_PROXY_PDF_NAME} (merged ' + str(len(sys.argv)-1) + ' files)')
except ImportError:
    import subprocess, shutil
    gs = shutil.which('gs')
    if gs:
        subprocess.run([gs, '-dBATCH', '-dNOPAUSE', '-q', '-sDEVICE=pdfwrite',
                        '-sOutputFile=${RUN_ROOT}/${SPECTRA_PROXY_PDF_NAME}'] + sys.argv[1:], check=True)
        print('[PDF] ${SPECTRA_PROXY_PDF_NAME} (merged via gs)')
    else:
        print('[WARN] Cannot merge proxy spectra PDFs: neither pypdf nor gs available')
" "${PROXY_PDFS[@]}" 2>&1
  fi
fi

# --- Spectra ECMWF consolidated PDF ---
if [[ ! -f "${RUN_ROOT}/${SPECTRA_ECMWF_PDF_NAME}" ]]; then
  ECMWF_PDFS=()
  for candidate_dir in \
    "${DATA_DIR}/spectra_ecmwf"* \
    "${DATA_DIR}/spectra_proxy10_subset_ecmwf"* \
    "${RUN_ROOT}/spectra_ecmwf"* \
    "${RUN_ROOT}/spectra_proxy10_subset_ecmwf"*; do
    [[ -d "${candidate_dir}" ]] || continue
    for pdf_file in "${candidate_dir}"/physical_models_spectra_*.pdf; do
      [[ -f "${pdf_file}" ]] || continue
      ECMWF_PDFS+=("${pdf_file}")
    done
    if [[ ${#ECMWF_PDFS[@]} -gt 0 ]]; then
      break
    fi
  done

  if [[ ${#ECMWF_PDFS[@]} -gt 0 ]]; then
    python3 -c "
import sys
try:
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for src in sys.argv[1:]:
        for page in PdfReader(src).pages:
            writer.add_page(page)
    with open('${RUN_ROOT}/${SPECTRA_ECMWF_PDF_NAME}', 'wb') as f:
        writer.write(f)
    print('[PDF] ${SPECTRA_ECMWF_PDF_NAME} (merged ' + str(len(sys.argv)-1) + ' files)')
except ImportError:
    import subprocess, shutil
    gs = shutil.which('gs')
    if gs:
        subprocess.run([gs, '-dBATCH', '-dNOPAUSE', '-q', '-sDEVICE=pdfwrite',
                        '-sOutputFile=${RUN_ROOT}/${SPECTRA_ECMWF_PDF_NAME}'] + sys.argv[1:], check=True)
        print('[PDF] ${SPECTRA_ECMWF_PDF_NAME} (merged via gs)')
    else:
        print('[WARN] Cannot merge ECMWF spectra PDFs: neither pypdf nor gs available')
" "${ECMWF_PDFS[@]}" 2>&1
  fi
fi

# --- TC contour consolidated PDF ---
# Merge all_regions_plots.pdf from all tc_contour/tc_local directories into one
if [[ ! -f "${RUN_ROOT}/${TC_CONTOUR_PDF_NAME}" ]]; then
  TC_CONTOUR_PDFS=()
  for candidate_dir in \
    "${DATA_DIR}"/tc_contour_plots* \
    "${DATA_DIR}"/tc_local_plots* \
    "${RUN_ROOT}"/tc_contour_plots* \
    "${RUN_ROOT}"/tc_local_plots*; do
    [[ -d "${candidate_dir}" ]] || continue
    combined="${candidate_dir}/all_regions_plots.pdf"
    if [[ -f "${combined}" ]]; then
      TC_CONTOUR_PDFS+=("${combined}")
    fi
  done

  if [[ ${#TC_CONTOUR_PDFS[@]} -eq 1 ]]; then
    cp "${TC_CONTOUR_PDFS[0]}" "${RUN_ROOT}/${TC_CONTOUR_PDF_NAME}"
    echo "[PDF] ${TC_CONTOUR_PDF_NAME} (from $(basename "$(dirname "${TC_CONTOUR_PDFS[0]}")")/all_regions_plots.pdf)"
  elif [[ ${#TC_CONTOUR_PDFS[@]} -gt 1 ]]; then
    python3 -c "
import sys
try:
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for src in sys.argv[1:]:
        for page in PdfReader(src).pages:
            writer.add_page(page)
    with open('${RUN_ROOT}/${TC_CONTOUR_PDF_NAME}', 'wb') as f:
        writer.write(f)
    print('[PDF] ${TC_CONTOUR_PDF_NAME} (merged ' + str(len(sys.argv)-1) + ' TC contour files)')
except ImportError:
    import subprocess, shutil
    gs = shutil.which('gs')
    if gs:
        subprocess.run([gs, '-dBATCH', '-dNOPAUSE', '-q', '-sDEVICE=pdfwrite',
                        '-sOutputFile=${RUN_ROOT}/${TC_CONTOUR_PDF_NAME}'] + sys.argv[1:], check=True)
        print('[PDF] ${TC_CONTOUR_PDF_NAME} (merged via gs)')
    else:
        print('[WARN] Cannot merge TC contour PDFs: neither pypdf nor gs available')
" "${TC_CONTOUR_PDFS[@]}" 2>&1
  fi
fi

# --- TC PDF distribution ---
if [[ ! -f "${RUN_ROOT}/${TC_PDF_DIST_NAME}" ]]; then
  for tc_pdf in "${DATA_DIR}"/tc_normed_pdfs_*.pdf "${RUN_ROOT}"/tc_normed_pdfs_*.pdf; do
    [[ -f "${tc_pdf}" ]] || continue
    cp "${tc_pdf}" "${RUN_ROOT}/${TC_PDF_DIST_NAME}"
    echo "[PDF] ${TC_PDF_DIST_NAME} (from $(basename "${tc_pdf}"))"
    break
  done
fi

# --- TC member maps: one top-level PDF per event (not merged) ---
for candidate_dir in \
  "${DATA_DIR}"/tc_member_maps* \
  "${RUN_ROOT}"/tc_member_maps*; do
  [[ -d "${candidate_dir}" ]] || continue
  for pdf_file in "${candidate_dir}"/tc_members_*.pdf; do
    [[ -f "${pdf_file}" ]] || continue
    base="$(basename "${pdf_file}")"
    # Extract event name: tc_members_<event>_<label>_<date>.pdf -> <event>
    event="$(echo "${base}" | sed -n 's/^tc_members_\([^_]*\)_.*/\1/p')"
    if [[ -n "${event}" ]]; then
      top_name="tc_member_maps_${event}.pdf"
    else
      top_name="${base}"
    fi
    if [[ ! -f "${RUN_ROOT}/${top_name}" ]]; then
      cp "${pdf_file}" "${RUN_ROOT}/${top_name}"
      echo "[PDF] ${top_name} (from ${base})"
    fi
  done
  break
done

###############################################################################
# STEP 3: Canonical metrics.json assembly
###############################################################################

# If metrics.json doesn't exist yet, assemble it from available artifacts.
# This runs LAST so all eval jobs have already written their canonical outputs.
if [[ ! -f "${RUN_ROOT}/metrics.json" ]]; then
  DOWNSCALING_TOOLS_ROOT="${DOWNSCALING_TOOLS_ROOT:-/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools}"
  # Detect lane from EXPERIMENT_CONFIG.yaml if not set
  EFFECTIVE_LANE="${LANE}"
  if [[ -z "${EFFECTIVE_LANE}" ]] && [[ -f "${RUN_ROOT}/EXPERIMENT_CONFIG.yaml" ]]; then
    EFFECTIVE_LANE="$(python3 -c "
import yaml, sys
try:
    config = yaml.safe_load(open('${RUN_ROOT}/EXPERIMENT_CONFIG.yaml'))
    print(config.get('lane', ''))
except Exception:
    pass
" 2>/dev/null || true)"
  fi

  if [[ -n "${EFFECTIVE_LANE}" ]]; then
    python3 -c "
import sys; sys.path.insert(0, '${DOWNSCALING_TOOLS_ROOT}')
from eval.jobs import scoreboard_metrics as metrics
from pathlib import Path
import yaml, json

run_root = Path('${RUN_ROOT}')
config = {}
config_path = run_root / 'EXPERIMENT_CONFIG.yaml'
if config_path.exists():
    config = yaml.safe_load(config_path.read_text()) or {}

out = metrics.build_run_scoreboard_metrics(
    run_id=config.get('run_id', '${RUN_ID}'),
    lane='${EFFECTIVE_LANE}',
    checkpoint_id=config.get('checkpoint_id', ''),
    checkpoint_step=int(config.get('checkpoint_step', 0)),
    output_root=run_root.parent,
    sigma_run_id=config.get('sigma_run_id', config.get('run_id', '${RUN_ID}')),
    tc_stats_path=run_root / 'data' / 'tc',
    spectra_dir=run_root / 'data' / 'spectra',
    surface_json_path=run_root / 'data' / 'surface_loss.json',
    event_names=config.get('tc_events', None),
    scope=config.get('scope', {}),
)
(run_root / 'metrics.json').write_text(json.dumps(out, indent=2))
n_scalars = len(out.get('scalars', {}))
print(f'[METRICS] Wrote canonical metrics.json ({n_scalars} scalars)')
" 2>&1 || echo "[WARN] metrics.json assembly failed (non-fatal)"

    # Legacy compat symlinks
    if [[ -f "${RUN_ROOT}/metrics.json" ]]; then
      ln -sfn "metrics.json" "${RUN_ROOT}/scoreboard_metrics.json" 2>/dev/null || true
      if [[ -n "${EFFECTIVE_LANE}" ]]; then
        ln -sfn "metrics.json" "${RUN_ROOT}/${EFFECTIVE_LANE}_metrics.json" 2>/dev/null || true
      fi
    fi
  else
    echo "[INFO] Skipping metrics.json assembly: no lane detected"
  fi
fi

###############################################################################
# STEP 4: Summary
###############################################################################

echo ""
echo "=== Lean eval layout summary ==="
echo "Run root: ${RUN_ROOT}"
echo ""
echo "Top-level PDFs:"
for pdf in "${RUN_ROOT}"/*.pdf; do
  [[ -f "${pdf}" ]] || continue
  echo "  $(basename "${pdf}")"
done

echo ""
echo "Top-level PNGs (training diagnostics):"
for png in "${RUN_ROOT}"/*.png; do
  [[ -f "${png}" ]] || continue
  echo "  $(basename "${png}")"
done

echo ""
echo "Top-level YAML:"
for yaml in "${RUN_ROOT}"/*.yaml; do
  [[ -f "${yaml}" ]] || continue
  echo "  $(basename "${yaml}")"
done

echo ""
echo "data/ contents:"
ls -1 "${DATA_DIR}" 2>/dev/null | while read -r item; do
  echo "  data/${item}"
done

echo ""
echo "DONE: lean eval layout finalized for ${RUN_ID}"
