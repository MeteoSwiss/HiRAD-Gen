#!/bin/bash
# Build an IFS-HRES trajectories anemoi zarr from precomputed GRIBs. Override RECIPE/ZARR via env:
#   sbatch --export=ALL,RECIPE=...,ZARR=... src/hirad/input_data/build_ifs_hres_full.sh
#SBATCH --job-name=ifs_build_full
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=/users/mmcgloho/hirad-gen/HiRAD-Gen/logs/ifs_build_full.log
#SBATCH -A c38
set -euo pipefail
cd /users/mmcgloho/hirad-gen/HiRAD-Gen
RECIPE=${RECIPE:-src/hirad/input_data/configs/ifs_hres_train_full.yaml}
ZARR=${ZARR:-/capstor/scratch/cscs/mmcgloho/ifs-hres-realch1/ifs_hres_traj_202503_202608_1h_90step.zarr}
srun --environment=/users/mmcgloho/hirad-gen/HiRAD-Gen/ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-deps
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
