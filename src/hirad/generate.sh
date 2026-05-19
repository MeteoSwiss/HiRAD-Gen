#!/bin/bash
# Generate predictions.
#
# Default mode (no SLURM array): single run using config defaults.
# Monthly array mode (SLURM_ARRAY_TASK_ID and START_MONTH set):
#   each array task generates one month, starting from START_MONTH
#   (array index 0 -> START_MONTH, index 1 -> next month, ...).
#   Submit via ./src/hirad/submit_monthly.sh START_MONTH END_MONTH.

#SBATCH --job-name="generate"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/regression_generation_%A_%a.log

### ENVIRONMENT ####
#SBATCH -A a161

set -euo pipefail

# Optional Hydra overrides for monthly array mode.
EXTRA_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${START_MONTH:-}" ]]; then
    MONTH="${START_MONTH}-01 +${SLURM_ARRAY_TASK_ID} months"
    NEXT_MONTH="${START_MONTH}-01 +$((SLURM_ARRAY_TASK_ID + 1)) months"
    START=$(date -u -d "$MONTH" +%Y%m%d-%H%M)
    END=$(date -u -d "$NEXT_MONTH" +%Y%m%d-%H%M)
    TAG=$(date -u -d "$MONTH" +%Y_%m)
    echo "Generating ${TAG}: ${START} -> ${END}"
    EXTRA_ARGS+=(
        "generation.times_range=[${START},${END},1]"
        "hydra.run.dir=./outputs/generation/era_real_${TAG}"
    )
fi

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
echo "Master node : $MASTER_ADDR"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
echo "Master address : $MASTER_ADDR"
export MASTER_ADDR
export MASTER_PORT=29500
echo "Master port: $MASTER_PORT"

export OMP_NUM_THREADS=1

EXTRA_ARGS_STR="${EXTRA_ARGS[*]@Q}"
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e .
    python src/hirad/inference/generate.py --config-name=generate_era_real.yaml ${EXTRA_ARGS_STR}
"
