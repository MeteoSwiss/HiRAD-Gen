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
#SBATCH -A a0269

# Usage:
#   [CONFIG_NAME=src/hirad/conf/eval_cosmo.yaml] sbatch src/hirad/snapshots.sh \
#       [--inference-output-dir DIR] [YYYYMMDD-HHMM ...]
# CONFIG_NAME defaults to eval_real.yaml; DIR overrides inference_output_dir in it; the optional
# timesteps override the times in the config.
CONFIG_NAME=${CONFIG_NAME:-src/hirad/conf/eval_real.yaml}
EXTRA_ARGS=()
if [[ ${1:-} == --inference-output-dir ]]; then
    if [[ $# -lt 2 ]]; then
        echo "ERROR: --inference-output-dir needs a directory." >&2
        exit 1
    fi
    EXTRA_ARGS+=("--inference-output-dir" "$2")
    shift 2
fi
if [ $# -gt 0 ]; then
    EXTRA_ARGS+=("--times" "$@")
fi

EXTRA_ARGS_STR="${EXTRA_ARGS[*]@Q}"
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/eval/snapshots.py --config-name=${CONFIG_NAME} ${EXTRA_ARGS_STR}
"