#!/bin/bash

#SBATCH --job-name="ifso1280-real-diffusion-patched"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/train_ifso1280_real_diffusion_patched.log
#SBATCH --error=./logs/train_ifso1280_real_diffusion_patched.err

### ENVIRONMENT ####
#SBATCH -A c38

# Choose method to initialize dist in pythorch
export DISTRIBUTED_INITIALIZATION_METHOD=SLURM

# Get master node.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Get IP for hostname.
MASTER_ADDR="$(getent ahosts "$MASTER_ADDR" | awk '{ print $1; exit }')"
export MASTER_ADDR
export MASTER_PORT=29500

export OMP_NUM_THREADS=1

# Only local rank 0 installs - 4 concurrent `pip install -e .` per node (one per task)
# race on the same node-local site-packages otherwise. Other local ranks wait on the
# lock file rather than installing themselves.
srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    LOCK=/tmp/hirad_pip_install_done
    if [ \"\$SLURM_LOCALID\" = '0' ]; then
        rm -f \"\$LOCK\"
        pip install -e . --no-dependencies
        touch \"\$LOCK\"
    else
        while [ ! -f \"\$LOCK\" ]; do sleep 1; done
    fi
    python src/hirad/training/train.py --config-name=training_ifso1280_real_diffusion_patched.yaml
"
