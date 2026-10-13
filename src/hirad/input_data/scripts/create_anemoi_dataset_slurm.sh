#!/bin/bash
# SLURM wrapper around create_anemoi_dataset.sh, using the same container
# environment as training (ci/edf/modulus_env.toml). See the
# anemoi-dataset-create skill for background.
#
# CAVEAT -- unverified: MARS retrieval needs outbound internet access to
# ECMWF's web API. We've only confirmed that works from the Alps login node
# (see create_anemoi_dataset.sh); whether compute nodes have that route has
# not been tested. If this job hangs/errors specifically on MARS retrieval
# (not on `pip install` or `init`'s own setup), that's the likely cause --
# fall back to running create_anemoi_dataset.sh directly on the login node.
#
# Usage:
#   RECIPE=... OUTPUT=... N_PARTS=... sbatch create_anemoi_dataset_slurm.sh
# or edit the defaults below and just:
#   sbatch src/hirad/input_data/scripts/create_anemoi_dataset_slurm.sh

#SBATCH --job-name="anemoi-dataset-create"

### HARDWARE ###
# CPU/network bound, no GPU needed -- unlike training jobs this doesn't
# need --exclusive or gpus-per-node.
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --no-requeue

### OUTPUT ###
#SBATCH --output=./logs/create_anemoi_dataset_%j.log

### ENVIRONMENT ###
#SBATCH -A a161

set -euo pipefail

RECIPE=${RECIPE:-src/hirad/input_data/configs/medrange.yaml}
OUTPUT=${OUTPUT:-/capstor/scratch/cscs/mmcgloho/anemoitest/leadtime1h-small-0p1degree.zarr}
N_PARTS=${N_PARTS:-12}
LOG_DIR=${LOG_DIR:-./logs/$(basename "$OUTPUT" .zarr)}

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies
    src/hirad/input_data/scripts/create_anemoi_dataset.sh '$RECIPE' '$OUTPUT' '$N_PARTS' '$LOG_DIR'
"
