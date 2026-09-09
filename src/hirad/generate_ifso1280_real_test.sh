#!/bin/bash

#SBATCH --job-name="ifso1280-real-generate-test"

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
#SBATCH --output=./logs/generate_ifso1280_real_test.log

### ENVIRONMENT ####
#SBATCH -A c38

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=1

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    LOCK=/tmp/hirad_pip_install_done
    if [ \"\$SLURM_LOCALID\" = '0' ]; then
        rm -f \"\$LOCK\"
        pip install -e . --no-dependencies
        touch \"\$LOCK\"
    else
        while [ ! -f \"\$LOCK\" ]; do sleep 1; done
    fi
    python src/hirad/inference/generate.py --config-name=generate_ifso1280_real.yaml
"
