#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --expver <expver> [options]

Options:
  --expver <id>               Required experiment version (e.g. j24v)
  --eval-root <path>          Eval root (default: /home/ecm5702/scratch/eval)
  --eval-date <A/B>           MARS eval date range for predictions.nc (default: 20230801/20230802)
  --eval-number <list>        Members for eval.run mars-expver (default: 1/2)
  --eval-step <list>          Steps for eval.run mars-expver (default: 24/120)
  --quaver-first-date <d>     Quaver first reference date YYYYMMDD (default: 20230826)
  --quaver-last-date <d>      Quaver last reference date YYYYMMDD (default: 20230827)
  --quaver-nmem <n>           Quaver nmem (default: 2)
  --hres-grid <grid>          High-resolution grid tag for Quaver/spectra/TC (default: O320)
  --hres-reference-grib <f>   High-resolution reference grib for eval.run mars-expver (default: enfo_reference_o320-early-august.grib)
  --spectra-hres-ref <name>   High-resolution spectra reference folder (default: enfo_o320)
  --spectra-hres-label <txt>  High-resolution spectra legend label (default: enfo O320)
  --tc-exp-prefix <prefix>    TC ML exp prefix (default: ENFO_<hres-grid>, e.g. ENFO_O320)
  --spectra-date <mars>       Spectra MARS date range (default: 20230826/to/20230827/by/1)
  --spectra-step <h>          Spectra step (default: 144)
  --dry-run                   Only generate scripts, do not submit
EOF
}

EXPVER=""
EVAL_ROOT="/home/ecm5702/scratch/eval"
EVAL_DATE="20230801/20230802"
EVAL_NUMBER="1/2"
EVAL_STEP="24/120"
QUAVER_FIRST_DATE="20230826"
QUAVER_LAST_DATE="20230827"
QUAVER_NMEM="2"
HRES_GRID="O320"
HRES_REFERENCE_GRIB="enfo_reference_o320-early-august.grib"
SPECTRA_HRES_REF_NAME="enfo_o320"
SPECTRA_HRES_LABEL="enfo O320"
TC_EXP_PREFIX=""
SPECTRA_DATE="20230826/to/20230827/by/1"
SPECTRA_STEP="144"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expver) EXPVER="$2"; shift 2;;
    --eval-root) EVAL_ROOT="$2"; shift 2;;
    --eval-date) EVAL_DATE="$2"; shift 2;;
    --eval-number) EVAL_NUMBER="$2"; shift 2;;
    --eval-step) EVAL_STEP="$2"; shift 2;;
    --quaver-first-date) QUAVER_FIRST_DATE="$2"; shift 2;;
    --quaver-last-date) QUAVER_LAST_DATE="$2"; shift 2;;
    --quaver-nmem) QUAVER_NMEM="$2"; shift 2;;
    --hres-grid) HRES_GRID="$2"; shift 2;;
    --hres-reference-grib) HRES_REFERENCE_GRIB="$2"; shift 2;;
    --spectra-hres-ref) SPECTRA_HRES_REF_NAME="$2"; shift 2;;
    --spectra-hres-label) SPECTRA_HRES_LABEL="$2"; shift 2;;
    --tc-exp-prefix) TC_EXP_PREFIX="$2"; shift 2;;
    --spectra-date) SPECTRA_DATE="$2"; shift 2;;
    --spectra-step) SPECTRA_STEP="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$EXPVER" ]]; then
  echo "--expver is required" >&2
  usage
  exit 2
fi

HRES_GRID_LOWER="${HRES_GRID,,}"
if [[ -z "${TC_EXP_PREFIX}" ]]; then
  TC_EXP_PREFIX="ENFO_${HRES_GRID}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POST_PREPML_ROOT="${POST_PREPML_ROOT:-/home/ecm5702/dev/post_prepml}"
RUN_DIR="${EVAL_ROOT}/${EXPVER}"
JOBS_DIR="${RUN_DIR}/jobs"
mkdir -p "${JOBS_DIR}" "${RUN_DIR}/logs"

cat > "${JOBS_DIR}/eval_mars_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=eval_mars_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=${RUN_DIR}/logs/eval_mars_${EXPVER}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
python -m eval.run --eval-root ${EVAL_ROOT} mars-expver --expver ${EXPVER} --run-name ${EXPVER} --date ${EVAL_DATE} --number ${EVAL_NUMBER} --step ${EVAL_STEP} --high-res-reference-grib ${HRES_REFERENCE_GRIB}
EOF

cat > "${JOBS_DIR}/quaver_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=quaver_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=${RUN_DIR}/logs/quaver_${EXPVER}_%j.out
set -euo pipefail
module load quaver
QUAVER_OUT_DIR="${RUN_DIR}/quaver"
mkdir -p "\${QUAVER_OUT_DIR}"
cd "\${QUAVER_OUT_DIR}"
START_TS=\$(date +%s)
quaver ${PROJECT_ROOT}/eval/quaver/q_compute_probabilistic.py \\
  --expver ${EXPVER} \\
  --nmem ${QUAVER_NMEM} \\
  --first_reference_date ${QUAVER_FIRST_DATE} \\
  --last_reference_date ${QUAVER_LAST_DATE} \\
  --date_step 24 \\
  --first_lead_time 24 \\
  --last_lead_time 120 \\
  --lead_time_step 24 \\
  --grid ${HRES_GRID} --class rd --database fdb
# Quaver plot scripts can write quaver.pdf in CWD or HOME depending on runtime defaults.
# Always consolidate any produced PDFs in the main eval run folder.
for cand in "\${QUAVER_OUT_DIR}"/*.pdf; do
  [ -f "\${cand}" ] || continue
  cp -f "\${cand}" "${RUN_DIR}/"
done
HOME_QUAVER="/etc/ecmwf/nfs/dh2_home_a/ecm5702/quaver.pdf"
if [ -f "\${HOME_QUAVER}" ]; then
  PDF_TS=\$(stat -c %Y "\${HOME_QUAVER}" 2>/dev/null || echo 0)
  if [ "\${PDF_TS}" -ge "\${START_TS}" ]; then
    cp -f "\${HOME_QUAVER}" "${RUN_DIR}/quaver_${EXPVER}.pdf"
  fi
fi
EOF

cat > "${JOBS_DIR}/spectra_compute_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=spectra_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=${RUN_DIR}/logs/spectra_${EXPVER}_%j.out
set -euo pipefail
module unload ecmwf-toolbox || true
module load ifs
expid="${EXPVER}"
thegrid="${HRES_GRID_LOWER}"
thedate="${SPECTRA_DATE}"
thestep="${SPECTRA_STEP}"
outbase="/home/ecm5702/perm/ai_spectra/\${expid}"
mkdir -p "\${outbase}/grb" "\${outbase}/spectral_harmonics" "\${outbase}/spectra"
params=("2t:sfc" "10u:sfc" "10v:sfc" "sp:sfc" "t:850" "z:500")
for param_level in "\${params[@]}"; do
  param="\${param_level%%:*}"
  level="\${param_level##*:}"
  outdir="\${outbase}/grb/\${param}_\${level}"
  mkdir -p "\${outdir}"
  if [ "\${level}" = "sfc" ]; then
mars <<MARS
retrieve,
 type=pf,date=\${thedate},levtype=sfc,param=\${param},step=\${thestep},time=0000,number=1/2,
 grid=\${thegrid},stream=enfo,class=rd,expver=\${expid},area=89/0/-89/360,
 target="\${outdir}/[expver]_[date]_[step]_[number]_nopoles.grb"
MARS
  else
mars <<MARS
retrieve,
 type=pf,date=\${thedate},levtype=pl,level=\${level},param=\${param},step=\${thestep},time=0000,number=1/2,
 grid=\${thegrid},stream=enfo,class=rd,expver=\${expid},area=89/0/-89/360,
 target="\${outdir}/[expver]_[date]_[step]_[number]_nopoles.grb"
MARS
  fi
done
module unload ecmwf-toolbox || true
module load eclib
module load pifsenv
module load ifs
export DR_HOOK_ASSERT_MPI_INITIALIZED=0
indir="\${outbase}/grb"
wdir="\${outbase}/spectral_harmonics"
mkdir -p "\${wdir}"
param_dirs=("2t_sfc" "10u_sfc" "10v_sfc" "sp_sfc" "t_850" "z_500")
for param_dir in "\${param_dirs[@]}"; do
  mkdir -p "\${wdir}/\${param_dir}"
  for xfile in \${indir}/\${param_dir}/\${expid}_*.grb; do
    [ -f "\${xfile}" ] || continue
    bname=\$(basename "\${xfile}")
    gptosp.ser -l -g "\${xfile}" -S "\${wdir}/\${param_dir}/\${bname}_sh"
  done
done
module unload ifs || true
module load ecmwf-toolbox
source /home/ecm5702/dev/.ds-dyn/bin/activate
python /tmp/compute_spectra_\${expid}_lite.py 2>/dev/null || true
cp ${POST_PREPML_ROOT}/spectra/spectra_ml/individual_files/compute_spectra-3.py /tmp/compute_spectra_\${expid}_lite.py
sed -i 's/2023-08-01 00:00:00/2023-08-26 00:00:00/g; s/2023-08-10 00:00:00/2023-08-27 00:00:00/g' /tmp/compute_spectra_\${expid}_lite.py
python /tmp/compute_spectra_\${expid}_lite.py --expver \${expid}
EOF

cat > "${JOBS_DIR}/spectra_plot_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=specplot_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=${RUN_DIR}/logs/specplot_${EXPVER}_%j.out
set -euo pipefail
module unload ifs || true
module load ecmwf-toolbox
source /home/ecm5702/dev/.ds-dyn/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
python -m eval._backends.spectra.plot_spectra_compare --expver ${EXPVER} --date-start "2023-08-26 00:00:00" --date-end "2023-08-26 00:00:00" --steps ${SPECTRA_STEP} --members 1 --output-dir ${RUN_DIR} --hres-reference-name ${SPECTRA_HRES_REF_NAME} --hres-reference-label "${SPECTRA_HRES_LABEL}"
EOF

cat > "${JOBS_DIR}/tc_retrieve_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=tcget_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=${RUN_DIR}/logs/tc_get_${EXPVER}_%j.out
set -euo pipefail
STREAM="ENFO"; GRID="${HRES_GRID}"; EXPID="${EXPVER}"; STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"; NUMBER="1/to/10/by/1"; CLASS="RD"; TYPE="PF"; LEVEL="SFC"; DATABASE="FDB"
fetch_event () {
  local event="\$1"; local area="\$2"; shift 2; local days=("\$@"); local dir="/home/ecm5702/perm/reference/o96_o320/tc/\${event}"; mkdir -p "\${dir}"
  for d in "\${days[@]}"; do
    date="2023""08"\${d}; target="\${dir}/surface_pf_\${STREAM}_\${GRID}_\${EXPID}_\${date}.grib"
    [ -s "\${target}" ] && continue
mars <<MARS
RETRIEVE,
  CLASS=\${CLASS},TYPE=\${TYPE},STREAM=\${STREAM},EXPVER=\${EXPID},LEVTYPE=\${LEVEL},AREA=\${area},PARAM=\${PARAMS},
  DATE=\${date},TIME=0000,NUMBER=\${NUMBER},GRID=\${GRID},DATABASE=\${DATABASE},STEP=\${STEP},TARGET="\${target}"
MARS
  done
}
fetch_event idalia   "40/-100/10/-70" 26 27 28 29 30
fetch_event hilary   "35/-125/0/-95" 16 17 18 19 20
fetch_event franklin "40/-80/10/-50" 20 21 22 23 24 25 26 27 28 29 30
fetch_event fernanda "30/-135/0/-105" 12 13 14 15 16 17
fetch_event dora     "25/175/5/-105" 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17
EOF

cat > "${JOBS_DIR}/tc_plot_${EXPVER}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=tcplot_${EXPVER}
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=${RUN_DIR}/logs/tc_plot_${EXPVER}_%j.out
set -euo pipefail
source /home/ecm5702/dev/.ds-dyn/bin/activate
module unload ifs || true
module load ecmwf-toolbox
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"
python -m eval._backends.tc.workflows legacy-members --expver ${EXPVER} --outdir ${RUN_DIR}
python -m eval._backends.tc.workflows pdf --outdir ${RUN_DIR} --out-name tc_normed_pdfs_all_events_${EXPVER}.pdf --ml-expids ${TC_EXP_PREFIX}_${EXPVER}
EOF

chmod +x "${JOBS_DIR}"/*.sbatch

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run. Scripts written to ${JOBS_DIR}"
  exit 0
fi

jid_eval=$(sbatch "${JOBS_DIR}/eval_mars_${EXPVER}.sbatch" | awk '{print $NF}')
jid_quaver=$(sbatch "${JOBS_DIR}/quaver_${EXPVER}.sbatch" | awk '{print $NF}')
jid_spectra=$(sbatch "${JOBS_DIR}/spectra_compute_${EXPVER}.sbatch" | awk '{print $NF}')
jid_specplot=$(sbatch --dependency=afterok:${jid_spectra} "${JOBS_DIR}/spectra_plot_${EXPVER}.sbatch" | awk '{print $NF}')
jid_tc_get=$(sbatch "${JOBS_DIR}/tc_retrieve_${EXPVER}.sbatch" | awk '{print $NF}')
jid_tc_plot=$(sbatch --dependency=afterok:${jid_tc_get} "${JOBS_DIR}/tc_plot_${EXPVER}.sbatch" | awk '{print $NF}')

cat <<EOF
Submitted:
  eval_mars      ${jid_eval}
  quaver         ${jid_quaver}
  spectra        ${jid_spectra}
  spectra_plot   ${jid_specplot} (afterok:${jid_spectra})
  tc_retrieve    ${jid_tc_get}
  tc_plot        ${jid_tc_plot} (afterok:${jid_tc_get})

Monitor:
  squeue -j ${jid_eval},${jid_quaver},${jid_spectra},${jid_specplot},${jid_tc_get},${jid_tc_plot}
EOF
