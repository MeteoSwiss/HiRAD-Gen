#!/bin/bash
# Build the IFS-HRES trajectories anemoi zarr for the 2025-04..2026-03 test period from
# precomputed GRIBs. Override RECIPE/ZARR via env:
#   sbatch --export=ALL,RECIPE=...,ZARR=... src/hirad/input_data/build_ifs_hres_202504_202603.sh
#
# Adapted from build_ifs_hres_full.sh (pstamenk) for this repo checkout/account.
#SBATCH --job-name=ifs_build_202504_202603
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=./logs/ifs_build_202504_202603.log
#SBATCH -A c38
set -euo pipefail
cd /users/mmcgloho/hirad-gen/HiRAD-Gen
RECIPE=${RECIPE:-src/hirad/input_data/configs/ifs_hres_traj_202504_202603.yaml}
ZARR=${ZARR:-/capstor/scratch/cscs/mmcgloho/ifs-hres-realch1/ifs_hres_traj_202504_202603.zarr}
srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    # Container's baked-in anemoi-datasets lacks anemoi.datasets.create.arguments
    # (needed by anemoi_sources.py's ForecastDates import) -- pin to the version
    # confirmed to have it.
    pip install anemoi-datasets==0.5.42
    python -m hirad.input_data.build_ifs_hres_anemoi --recipe '${RECIPE}' --path '${ZARR}' --overwrite
    echo '===== VERIFY ====='
    python - <<'PY'
from anemoi.datasets import open_dataset
ds = open_dataset('${ZARR}')
print('shape', ds.shape, 'variables', list(ds.variables))
print('base_dates', len(ds.base_dates), ds.base_dates[0], '->', ds.base_dates[-1])
print('steps', len(ds.steps))
PY
"
