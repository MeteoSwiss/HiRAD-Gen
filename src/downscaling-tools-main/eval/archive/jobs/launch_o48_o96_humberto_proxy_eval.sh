#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_INPUT_ROOT="/home/ecm5702/hpcperm/data/input_data/o48_o96/humberto_20250926_20250930"
DEFAULT_PROXY_BUNDLE_PAIRS="20250926:24,20250926:48,20250927:24,20250927:48,20250928:24,20250928:48,20250929:24,20250929:48,20250930:24,20250930:48"
DEFAULT_TC_EVENTS="humberto"
DEFAULT_SPECTRA_WEATHER_STATES="10u,10v,2t,msl,t_850,z_500"

# For strict rebuilt truth-aware bundles, pass:
#   --input-root /home/ecm5702/scratch/eval/<RUN_ID>/bundles_with_y
# The wrapper keeps the historical Humberto source tree as the default input root
# so existing prebuilt runs remain reproducible.

exec "${SCRIPT_DIR}/launch_proxy_eval.sh" \
  --input-root "${DEFAULT_INPUT_ROOT}" \
  --proxy-bundle-pairs "${DEFAULT_PROXY_BUNDLE_PAIRS}" \
  --proxy-n-files 10 \
  --scoreboard-subset-dir-name proxy10_humberto_subset \
  --scoreboard-spectra-dir-name spectra_humberto_proxy10 \
  --scoreboard-tc-events "${DEFAULT_TC_EVENTS}" \
  --scoreboard-tc-support-mode regridded \
  --scoreboard-spectra-weather-states "${DEFAULT_SPECTRA_WEATHER_STATES}" \
  --scoreboard-spectra-nside 64 \
  --scoreboard-spectra-lmax 95 \
  --scoreboard-spectra-member-agg per-file-mean \
  "$@"
