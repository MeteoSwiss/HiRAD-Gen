#!/bin/bash

set -euo pipefail

# Submits one job per evaluation script. Usage:
#   [CONFIG_NAME=src/hirad/conf/eval_cosmo.yaml] bash src/hirad/eval_wind.sh [--inference-output-dir DIR]
# CONFIG_NAME defaults to eval_real.yaml; DIR overrides inference_output_dir in it.

### CONFIG ###
CONFIG_NAME=${CONFIG_NAME:-src/hirad/conf/eval_real.yaml}
OUTPUT_DIR_ARG=""
if [[ $# -eq 2 && $1 == --inference-output-dir ]]; then
    OUTPUT_DIR_ARG=" --inference-output-dir=$2"
elif [[ $# -ne 0 ]]; then
    echo "Usage: [CONFIG_NAME=...] $0 [--inference-output-dir DIR]" >&2
    exit 1
fi

CMDS=(
    # Diurnal cycle of windspeed
    "python src/hirad/eval/diurnal_cycle_wind.py"
    # Probability of exceedance
    "python src/hirad/eval/probability_of_exceedance_wind.py"
    # QQ
    "python -m hirad.eval.bias_by_percentile_wind"
    "python -m hirad.eval.bias_by_percentile_wind_smoothed"
    # Maps
    "python src/hirad/eval/map_wind_stats.py"
    "python src/hirad/eval/map_wind_percentiles.py"
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
        -A a0269 \
        --output="./logs/plots_wind_${name}_%j.log" \
        --parsable \
        --wrap="srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -lc 'pip install -e . && ${cmd} --config-name=${CONFIG_NAME}${OUTPUT_DIR_ARG}'")
    echo "Submitted ${name}: ${job_id}"
done