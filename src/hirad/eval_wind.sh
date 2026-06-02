#!/bin/bash

set -euo pipefail

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

CMDS=(
    # Diurnal cycle of windspeed
    "python src/hirad/eval/diurnal_cycle_wind.py"
    # Probability of exceedance
    "python src/hirad/eval/probability_of_exceedance_wind.py"
    # Maps
    "python src/hirad/eval/map_wind_stats.py"
)

for cmd in "${CMDS[@]}"; do
    name=$(basename "${cmd##* }" .py | tr '.' '_')
    job_id=$(sbatch \
        --job-name="eval_wind_${name}" \
        --partition=normal \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --cpus-per-task=72 \
        --time=24:00:00 \
        --no-requeue \
        --exclusive \
        -A c38 \
        --output="./logs/plots_wind_${name}_%j.log" \
        --parsable \
        --wrap="srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -lc 'pip install -e . && ${cmd} --config-name=${CONFIG_NAME}'")
    echo "Submitted ${name}: ${job_id}"
done