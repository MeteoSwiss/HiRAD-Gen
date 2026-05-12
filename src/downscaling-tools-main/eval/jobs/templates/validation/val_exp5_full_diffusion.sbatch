#!/bin/bash --login
# ============================================================================
# Validation Exp 5: Full 78-var diffusion training on full dataset
#
# PURPOSE: Full-scale sanity check. All variables, full dataset,
#          probabilistic_low_noise, 5000 steps. Verifies the complete pipeline
#          trains stably with no NaN or divergence.
# PASS CRITERION: Loss monotonically decreasing, no NaN.
# ============================================================================

#SBATCH --job-name=val_exp5_full_diff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --qos=ng
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/jobscripts/outputs/%x_%j.out
#SBATCH --error=/home/ecm5702/dev/jobscripts/outputs/%x_%j.err

set -euxo pipefail

# ─── Environment ────────────────────────────────────────────────────────────
HOST_SHORT=$(hostname -s)
if [[ "$HOST_SHORT" == ag* ]]; then
  source /home/ecm5702/dev/.ds-ag/bin/activate
elif [[ "$HOST_SHORT" == ac* ]]; then
  source /home/ecm5702/dev/.ds-dyn/bin/activate
else
  echo "ERROR: Unknown host $HOST_SHORT" && exit 1
fi
export ANEMOI_BASE_SEED=756
export HYDRA_FULL_ERROR=1
export NCCL_IB_TIMEOUT=30
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATA_DIR=""
export DEV=/home/ecm5702/dev/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/

# ─── Diagnostics ────────────────────────────────────────────────────────────
hostname -s
echo "VIRTUAL_ENV=$VIRTUAL_ENV"
git -C /home/ecm5702/dev/anemoi-core branch --show-current
which python
python -V
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home/ecm5702/dev

RUN_NAME="val_exp5_full_diff_${SLURM_JOB_ID}"

# ─── Train ──────────────────────────────────────────────────────────────────
srun anemoi-training train \
  --config-name=o48_o96 \
  dataloader=o48_o96 \
  diagnostics.log.mlflow.run_name="${RUN_NAME}" \
  hardware.num_gpus_per_model=1 \
  hardware.num_gpus_per_node=1 \
  training.max_steps=5000 \
  training.lr.rate=0.001 \
  training.lr.iterations=5000 \
  training.lr.warmup=100 \
  training.lr.min=1e-5 \
  training.training_approach=probabilistic_low_noise \
  model.model.diffusion.log_normal_mean=0 \
  model.model.diffusion.log_normal_std=3.5 \
  dataloader.batch_size.training=1 \
  '+diagnostics.check_val_every_n_epoch=100' \
  '~training.variable_groups.pl.param' \
  '+training.variable_groups.pl.is_pressure_level=True'
