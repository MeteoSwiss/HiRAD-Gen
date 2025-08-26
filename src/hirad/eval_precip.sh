#!/bin/bash

#SBATCH --job-name="eval_precip"

### HARDWARE ###
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --exclusive

### OUTPUT ###
#SBATCH --output=./logs/plots_precip.log

### ENVIRONMENT ####
#SBATCH -A a161

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

# Get number of physical cores using Python
# PHYSICAL_CORES=$(python -c "import psutil; print(psutil.cpu_count(logical=False))")
# # Use SLURM_NTASKS (number of processes to be launched by torchrun)
# LOCAL_PROCS=${SLURM_NTASKS_PER_NODE:-1}
# # Compute threads per process
# OMP_THREADS=$(( PHYSICAL_CORES / LOCAL_PROCS ))
# export OMP_NUM_THREADS=$OMP_THREADS
export OMP_NUM_THREADS=72
# echo "Physical cores: $PHYSICAL_CORES"
# echo "Local processes: $LOCAL_PROCS"
# echo "Setting OMP_NUM_THREADS=$OMP_NUM_THREADS"


srun --environment=./ci/edf/modulus_env.toml bash -c "
    pip install -e . --no-dependencies

    # Diurnal cycle
    python src/hirad/eval/diurnal_cycle_precip_mean_wet-hour.py --config-name=generate_era_cosmo.yaml
    python src/hirad/eval/diurnal_cycle_precip_p99.py --config-name=generate_era_cosmo.yaml
    python src/hirad/eval/diurnal_cycle_temp_wind.py --config-name=generate_era_cosmo.yaml # TODO: Transfer to relevant script.

    # Histograms
    python src/hirad/eval/hist.py --config-name=generate_era_cosmo.yaml
    python src/hirad/eval/probability_of_exceedance.py --config-name=generate_era_cosmo.yaml

    # Maps
    python src/hirad/eval/map_precip_stats.py --config-name=generate_era_cosmo.yaml
    # python src/hirad/eval/snapshots.py --config-name=generate_era_cosmo.yaml
"