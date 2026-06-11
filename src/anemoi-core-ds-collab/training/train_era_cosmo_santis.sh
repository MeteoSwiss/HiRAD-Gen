#!/bin/bash
#SBATCH --account=c38
#SBATCH -t 12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

export ANEMOI_BASE_SEED=1

srun bash -c "source /users/mmcgloho/hirad-gen/HiRAD-Gen/.venv/bin/activate && anemoi-training train --config-name=era_cosmo_downscaling_santis"
