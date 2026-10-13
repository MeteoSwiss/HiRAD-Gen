#!/bin/bash
# Quick smoke test for torch_to_grib.py: converts one base_time from
# gen_ifs_ft_new into the user's own scratch dir, to sanity-check the FIS/z
# GRIB fix on real data before running the full March 2025 batch.

#SBATCH --job-name="torch-to-grib-test"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=./logs/torch_to_grib_test.log

### ENVIRONMENT ####
#SBATCH -A a0269

set -euo pipefail

SRC_DIR=/capstor/scratch/cscs/pstamenk/outputs/generation/gen_ifs_ft_new
DST_DIR=/capstor/scratch/cscs/mmcgloho/outputs/generation/gen_ifs_ft_new/grib

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-deps
    pip install anemoi-datasets==0.5.42
    python -m hirad.utils.torch_to_grib \
        ${SRC_DIR} all \
        --base-time 20250301-0000 \
        --dataset-cfg src/hirad/conf/dataset/anemoi_ifso1280_real_inference.yaml \
        --grib-out-dir ${DST_DIR}
"
