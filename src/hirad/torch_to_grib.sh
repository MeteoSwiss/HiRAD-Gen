#!/bin/bash
# Regenerate GRIB output (now including the FIS/z surface geopotential field)
# for March 2025 from gen_ifs_ft_new, writing into the user's own scratch dir
# rather than the original (pstamenk's) output tree.

#SBATCH --job-name="torch-to-grib-march2025"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=./logs/torch_to_grib_march2025_%j.log

### ENVIRONMENT ####
#SBATCH -A a0269

set -euo pipefail

SRC_DIR=/capstor/scratch/cscs/pstamenk/outputs/generation/gen_ifs_ft_new
DST_DIR=/capstor/scratch/cscs/mmcgloho/outputs/generation/gen_ifs_ft_new/grib_with_baseline


BASE_TIMES=""
for d in $(seq -w 1 31); do
    BASE_TIMES="${BASE_TIMES} 202507${d}-0000 202507${d}-1200"
done
#BASE_TIMES='all'

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-deps
    pip install anemoi-datasets==0.5.42
    python -m hirad.utils.torch_to_grib \
        ${SRC_DIR} all \
        --base-time ${BASE_TIMES} \
        --dataset-cfg src/hirad/conf/dataset/anemoi_ifso1280_real_inference.yaml \
        --grib-out-dir ${DST_DIR} \
        --skip-existing
"
