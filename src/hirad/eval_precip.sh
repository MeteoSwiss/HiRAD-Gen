#!/bin/bash

set -euo pipefail

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

CMDS=(
    # Diurnal cycle
    "python src/hirad/eval/diurnal_cycle_precip_mean_wet-hour.py"
    "python src/hirad/eval/diurnal_cycle_precip_high_percentiles.py"
    # Histograms
    "python src/hirad/eval/hist.py"
    "python src/hirad/eval/probability_of_exceedance.py"
    # QQ
    "python -m hirad.eval.bias_by_percentile_precip"
    # Maps
    "python src/hirad/eval/map_precip_stats.py"
    "python -m hirad.eval.diurnal_cycle_precip_maps"
)

for cmd in "${CMDS[@]}"; do
    name=$(basename "${cmd##* }" .py | tr '.' '_')
    job_id=$(sbatch \
        --job-name="eval_precip_${name}" \
        --partition=normal \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --cpus-per-task=72 \
        --time=24:00:00 \
        --no-requeue \
        --exclusive \
        -A c38 \
        --output="./logs/plots_precip_${name}_%j.log" \
        --parsable \
        --wrap="srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -lc 'pip install -e . && ${cmd} --config-name=${CONFIG_NAME}'")
    echo "Submitted ${name}: ${job_id}"
done