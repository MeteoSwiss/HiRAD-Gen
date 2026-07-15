#!/bin/bash
#SBATCH --account=c38
#SBATCH -t 12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4


export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

export ANEMOI_BASE_SEED=1

run_id=b24ed243e0504f98b9cf48396f871eac
checkpoint_dir=/capstor/scratch/cscs/mmcgloho/anemoi-downscaling/era-cosmo-run/checkpoints/$run_id/

for i in $(seq 99 100 800); do
  epoch_str=$(printf '%03d' $i)
  checkpoint_regex="anemoi-by_step-epoch_${epoch_str}-step_*.ckpt"
  filename=$(find $checkpoint_dir -type f -name "$checkpoint_regex" | head -n 1 | xargs basename)
  #cp $filename $checkpoint_dir/last.ckpt
  srun bash -c "source /users/mmcgloho/hirad-gen/HiRAD-Gen/.venv/bin/activate && \
    anemoi-training train \
    --config-name=era_cosmo_downscaling_santis \
    name=validation_full_set_epoch_$epoch_str \
    training.run_id=null \
    training.fork_run_id=$run_id \
    training.max_epochs=$(($i + 2)) \
    +hardware.paths.warm_start=$checkpoint_dir \
    hardware.files.warm_start=$filename \
    dataloader.limit_batches.training=1 \
    dataloader.limit_batches.validation=null \
    diagnostics.check_val_every_n_epoch=1"
done

#srun bash -c "source /users/mmcgloho/hirad-gen/HiRAD-Gen/.venv/bin/activate && \
#  anemoi-training train \
#  --config-name=era_cosmo_downscaling_santis \
#  training.run_id=null \
#  training.fork_run_id=b24ed243e0504f98b9cf48396f871eac \
##  training.max_epochs=801 \
 # dataloader.limit_batches.training=1 \
#  diagnostics.check_val_every_n_epoch=1"
