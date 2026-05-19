#!/bin/bash
# Submit monthly generation jobs over an inclusive month range as a SLURM array.
#
# Usage:
#   ./submit_monthly.sh START_MONTH END_MONTH [extra sbatch args...]
#
# Examples:
#   ./submit_monthly.sh 2021-01 2024-12       # 4 years
#   ./submit_monthly.sh 2022-06 2022-08       # JJA 2022

set -euo pipefail

usage() {
    echo "Usage: $0 START_MONTH END_MONTH [extra sbatch args...]" >&2
    echo "Example: $0 2022-06 2022-08" >&2
    exit 1
}

month_index() {
    local ym="$1"
    echo $(( 10#${ym%-*} * 12 + 10#${ym#*-} ))
}

(( $# >= 2 )) || usage

START_MONTH="$1" END_MONTH="$2"
shift 2

TASKS=$(( $(month_index "$END_MONTH") - $(month_index "$START_MONTH") + 1 ))
if (( TASKS <= 0 )); then
    echo "ERROR: END_MONTH (${END_MONTH}) is before START_MONTH (${START_MONTH})." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting ${TASKS} monthly jobs: ${START_MONTH} .. ${END_MONTH}"
exec sbatch \
    --array="0-$((TASKS - 1))" \
    --export=ALL,START_MONTH="${START_MONTH}",END_MONTH="${END_MONTH}" \
    --time=6:00:00 \
    --account=c38 \
    --output=./logs/generation_monthly_%A_%a.log \
    "$@" \
    "${SCRIPT_DIR}/generate.sh"
