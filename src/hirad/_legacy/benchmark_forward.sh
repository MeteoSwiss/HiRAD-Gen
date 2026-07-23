#!/bin/bash

#SBATCH --job-name="benchmark_forward"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/benchmark_forward.log

### ENVIRONMENT ###
#SBATCH -A c38

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

CKPT_PATH="/capstor/scratch/cscs/pstamenk/outputs/training/dit_era_real_cosine_lr_decay/checkpoints_diffusion_transformer"

srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -c "
    source ../hirad_new_env/bin/activate
    python benchmark_forward.py \
        --ckpt_path $CKPT_PATH \
        --batch_size 4 \
        --n_warmup 3 \
        --n_runs 20 \
        --compile   # uncomment to test torch.compile speedup
        # --fp16      # uncomment to test fp16
"
