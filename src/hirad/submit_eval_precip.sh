#!/bin/bash

CONFIG_NAME="src/hirad/conf/eval_real.yaml"

SCRIPTS=(
    "src/hirad/eval/diurnal_cycle_precip_mean_wet-hour.py"
    "src/hirad/eval/diurnal_cycle_precip_p99.py"
    "src/hirad/eval/hist.py"
    "src/hirad/eval/probability_of_exceedance.py"
    "src/hirad/eval/map_precip_stats.py"
)

for script in "${SCRIPTS[@]}"; do
    sbatch --export=ALL,EVAL_SCRIPT="$script",CONFIG_NAME="$CONFIG_NAME" src/hirad/eval_precip_job.sh
done
