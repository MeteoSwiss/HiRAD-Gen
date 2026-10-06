#!/bin/bash
# Compute Box-Cox(0.25) mean/std of tp over an anemoi dataset (input tp normalization stats).
# Override ZARR/CHANNEL/STRIDE via env, e.g.:
#   sbatch --export=ALL,ZARR=/path/to.zarr,STRIDE=8 src/hirad/input_data/calculate_tp_boxcox_stats.sh
#SBATCH --job-name=tp_boxcox_stats
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=/users/pstamenk/other/HiRAD-Gen/logs/tp_boxcox_stats.log
#SBATCH -A c38
set -euo pipefail
cd /users/pstamenk/other/HiRAD-Gen

ZARR=${ZARR:-/capstor/scratch/cscs/pstamenk/ifs-hres-realch1/ifs_hres_traj_2020_202502.zarr}
CHANNEL=${CHANNEL:-tp}
LMBDA=${LMBDA:-0.25}
STRIDE=${STRIDE:-8}
# Optional (gridded targets): restrict to a date range and open only the channel.
EXTRA=""
[[ -n "${START:-}" ]] && EXTRA="$EXTRA --start ${START}"
[[ -n "${END:-}" ]] && EXTRA="$EXTRA --end ${END}"
[[ "${SELECT:-0}" == "1" ]] && EXTRA="$EXTRA --select"

srun --environment=/users/pstamenk/other/HiRAD-Gen/ci/edf/modulus_env.toml bash -c "
    export PYTHONPATH=\${PWD}/src:\${PYTHONPATH:-}
    python -m hirad.input_data.calculate_tp_boxcox_stats \
        --zarr '${ZARR}' --channel '${CHANNEL}' --lmbda ${LMBDA} --stride ${STRIDE} ${EXTRA}
"
