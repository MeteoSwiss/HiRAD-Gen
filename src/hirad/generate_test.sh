#!/bin/bash
# Short test generation on the debug partition (2 GPUs). CONFIG selects the generate entry config
# (default generate_ifso1280_real), e.g. CONFIG=generate_era_real sbatch src/hirad/generate_test.sh

#SBATCH --job-name="corrdiff-test-generate"

### HARDWARE ###
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/generation_test.log

### ENVIRONMENT ####
#SBATCH -A a0269

CONFIG=${CONFIG:-generate_ifso1280_real}
CONFIG=${CONFIG%.yaml}

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

export OMP_NUM_THREADS=72

srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies
    python src/hirad/inference/generate.py --config-name=${CONFIG}.yaml
"