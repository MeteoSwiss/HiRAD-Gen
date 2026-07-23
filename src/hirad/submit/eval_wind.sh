#!/bin/bash

CONFIG_NAME="src/hirad/conf/eval_real.yaml"

SCRIPTS=(
    "src/hirad/eval/diurnal_cycle_temp_wind.py"
    "src/hirad/eval/probability_of_exceedance_wind.py"
    "src/hirad/eval/map_wind_stats.py"
)

for script in "${SCRIPTS[@]}"; do
    sbatch --export=ALL,EVAL_SCRIPT="$script",CONFIG_NAME="$CONFIG_NAME" src/hirad/eval_wind_job.sh
done
