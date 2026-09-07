#!/bin/bash
# Reusable IFS-HRES GRIB precompute over a date range. One SLURM array task per month; each
# task precomputes its whole month in a single pooled process (imports amortized, leads
# parallelised across cores). Only the full-3D 00Z/12Z runs are usable.
#
# IMPORTANT: set --array=0-(N_months-1) to match [START,END] (N_months inclusive of both ends).
# Override the range/paths via env, e.g.:
#   sbatch --export=ALL,START=2020-10-01,END=2025-02-28,OUT_DIR=/path,LEADS=1-33 \
#          --array=0-52%16 src/hirad/input_data/ifs_hres_precompute_full.sh
#SBATCH --job-name=ifs_pre_full
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --array=0-52%16
#SBATCH --output=/users/pstamenk/other/HiRAD-Gen/logs/ifs_pre_full_%A_%a.log
#SBATCH -A c38

set -euo pipefail
cd /users/pstamenk/other/HiRAD-Gen

START=${START:-2020-10-01}
END=${END:-2025-02-28}
LEADS=${LEADS:-1-33}
WORKERS=${WORKERS:-48}
OUT_DIR=${OUT_DIR:-/capstor/scratch/cscs/pstamenk/ifs-hres-realch1/precompute_2020_202502}
BASE=/capstor/store/mch/msopr/osm/IFS-HRES

# Ordered month list over [START, END] -> pick this task's month.
mapfile -t MONTHS < <(
  d="${START:0:7}-01"
  end_month="${END:0:7}-01"
  while [[ "$d" < "$end_month" || "$d" == "$end_month" ]]; do
    date -u -d "$d" +%Y-%m
    d=$(date -u -d "$d +1 month" +%Y-%m-01)
  done
)
echo "months in range: ${#MONTHS[@]} (set --array=0-$(( ${#MONTHS[@]} - 1 )))"
M=${MONTHS[$SLURM_ARRAY_TASK_ID]}          # YYYY-MM
YEAR_DIR="${BASE}/IFS-HRES${M:2:2}"        # IFS-HRES20 .. IFS-HRES25

# All 00/12Z init dirs in this month, clamped to [START, END].
month_start="${M}-01"; month_next=$(date -u -d "$month_start +1 month" +%Y-%m-01)
d="$month_start"; [[ "$d" < "$START" ]] && d="$START"
INIT_DIRS=()
while [[ "$d" < "$month_next" && ( "$d" < "$END" || "$d" == "$END" ) ]]; do
  ymd=$(date -u -d "$d" +%y%m%d)
  for c in 00 12; do INIT_DIRS+=("${YEAR_DIR}/${ymd}${c}"); done
  d=$(date -u -d "$d +1 day" +%Y-%m-%d)
done
echo "task ${SLURM_ARRAY_TASK_ID}: month ${M}, ${#INIT_DIRS[@]} init dirs -> ${OUT_DIR}"

srun --environment=/users/pstamenk/other/HiRAD-Gen/ci/edf/modulus_env.toml bash -c "
    export PYTHONPATH=\${PWD}/src:\${PYTHONPATH:-}
    export OMP_NUM_THREADS=1
    python -m hirad.input_data.ifs_hres_precompute \
        --init-dirs ${INIT_DIRS[*]} --out-dir '${OUT_DIR}' --leads ${LEADS} --workers ${WORKERS}
"
