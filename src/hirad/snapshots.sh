#!/bin/bash

#SBATCH --job-name="snapshots"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/snapshot.log

### ENVIRONMENT ####
#SBATCH -A c38

# Optional: pass specific timesteps as arguments (format: YYYYMMDD-HHMM)
# Usage: sbatch snapshots.sh 20230824-1400
EXTRA_ARGS=()
if [ $# -gt 0 ]; then
    EXTRA_ARGS+=("--times" "$@")
fi

EXTRA_ARGS_STR="${EXTRA_ARGS[*]@Q}"
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/eval/snapshots.py --config-name=src/hirad/conf/eval_real.yaml ${EXTRA_ARGS_STR}
"