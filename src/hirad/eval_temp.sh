#!/bin/bash

set -euo pipefail

### CONFIG ###
CONFIG_NAME="src/hirad/conf/eval_real.yaml"

CMDS=(
    # Diurnal cycle of 2m temperature
    "python src/hirad/eval/diurnal_cycle_temp.py"
    # QQ
    "python -m hirad.eval.bias_by_percentile_temp"
)

for cmd in "${CMDS[@]}"; do
    name=$(basename "${cmd##* }" .py | tr '.' '_')
    job_id=$(sbatch \
        --job-name="eval_temp_${name}" \
        --partition=normal \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --cpus-per-task=72 \
        --time=12:00:00 \
        --no-requeue \
        --exclusive \
        -A c38 \
        --output="./logs/plots_temp_${name}_%j.log" \
        --parsable \
        --wrap="srun --mpi=pmix --network=disable_rdzv_get --environment=./ci/edf/modulus_env.toml bash -lc 'pip install -e . && ${cmd} --config-name=${CONFIG_NAME}'")
    echo "Submitted ${name}: ${job_id}"
done
