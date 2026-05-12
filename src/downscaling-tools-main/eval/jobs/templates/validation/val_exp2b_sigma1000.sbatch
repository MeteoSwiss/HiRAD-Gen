#!/bin/bash --login
# ============================================================================
# Validation Exp 2b: 2t-only diffusion – FIXED σ=1000 diagnostic
#
# PURPOSE: Isolate whether the model CAN learn to denoise at a single noise
#          level. Uses probabilistic_low_noise with std≈0 so σ≈exp(0)=1.0
#          every step. If loss converges to ~0, the multi-σ optimization
#          (not the architecture) is the bottleneck. If not, the architecture
#          cannot denoise at all.
# PASS CRITERION: loss << 1.0 (the skip-connection baseline) within 2000 steps.
# ============================================================================

#SBATCH --job-name=val_exp2b_s1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --qos=ng
#SBATCH --mem=64G
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

RUN_NAME="val_exp2b_s1000_${SLURM_JOB_ID}"

# ─── Train ──────────────────────────────────────────────────────────────────
# log_normal_mean=0, log_normal_std=0.0001 → σ = exp(~0) ≈ 1.0 every step
# At σ=1, σ_d=1: c_skip=0.5, c_out=0.707, c_in=0.707 — balanced regime
srun anemoi-training train \
  --config-name=o48_o96 \
  data=downscaling_o48_2t_only \
  dataloader=o48_o96_2t_only \
  diagnostics.log.mlflow.run_name="${RUN_NAME}" \
  hardware.num_gpus_per_model=1 \
  hardware.num_gpus_per_node=1 \
  training.max_steps=2000 \
  training.lr.rate=0.001 \
  training.lr.iterations=2000 \
  training.lr.warmup=10 \
  training.lr.min=1e-5 \
  training.training_approach=deterministic \
  '+training.deterministic_sigma=1000.0' \
  '+dataloader.overfit_on_index=0' \
  dataloader.batch_size.training=1 \
  dataloader.num_workers.training=1 \
  dataloader.num_workers.validation=1 \
  dataloader.limit_batches.validation=1 \
  '+diagnostics.check_val_every_n_epoch=50' \
  '~training.variable_groups.pl.param' \
  '+training.variable_groups.pl.is_pressure_level=True'
