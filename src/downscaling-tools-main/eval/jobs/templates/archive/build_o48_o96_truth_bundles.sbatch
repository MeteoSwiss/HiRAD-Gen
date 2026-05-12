#!/bin/bash --login
# Canonical truth-aware bundle build for strict `o48 -> o96` manual inference.
# This stage rebuilds member-step bundle NetCDFs with explicit `target_hres_*`
# from the raw O48 input GRIBs and O96 target GRIBs before prediction.
# Typical Humberto flow:
#   1) edit SOURCE_GRIB_ROOT / RUN_ROOT / BUNDLE_DIR below
#   2) sbatch this template
#   3) then launch proxy/full prediction against INPUT_ROOT="${RUN_ROOT}/bundles_with_y"
#      (for example via launch_o48_o96_humberto_proxy_eval.sh --input-root "${RUN_ROOT}/bundles_with_y")
#SBATCH --job-name=o48_o96_truth_bundles
#SBATCH --qos=nf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/scratch/eval/_slurm/%x_%j.out

set -euo pipefail

###############################################################################
# USER SETTINGS (edit only this block)
###############################################################################
STACK_FLAVOR="new"         # new | old
SOURCE_HPC="ac"            # ac | ag | leonardo | jupiter
CHECKPOINT_SHORT="REPLACE_CHECKPOINT_SHORT"
SOURCE_GRIB_ROOT="/home/ecm5702/hpcperm/data/input_data/o48_o96/humberto_20250926_20250930"
RUN_ROOT="/home/ecm5702/scratch/eval/manual_example"
BUNDLE_DIR="/home/ecm5702/scratch/eval/manual_example/bundles_with_y"
DATES="20250926,20250927,20250928,20250929,20250930"
STEPS="24,48,72,96,120"
BUNDLE_PAIRS=""
MEMBERS="1"
HRES_STATIC_GRIB="${HRES_STATIC_GRIB:-}"   # optional override for hres z/lsm source
###############################################################################

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_bundle_pairs() {
  local raw="$1"
  python3 - "${raw}" <<'PY'
import sys

raw = sys.argv[1].strip()
if not raw:
    print("")
    raise SystemExit(0)

seen = set()
normalized = []
for token in raw.split(","):
    token = token.strip()
    if not token:
        continue
    if ":" not in token:
        raise SystemExit(
            "BUNDLE_PAIRS entries must use date:step format, for example 20250928:24."
        )
    date_raw, step_raw = token.split(":", 1)
    date_raw = date_raw.strip()
    step_raw = step_raw.strip()
    if len(date_raw) != 8 or not date_raw.isdigit():
        raise SystemExit(
            f"Invalid BUNDLE_PAIRS date {date_raw!r}; expected YYYYMMDD."
        )
    if not step_raw.isdigit():
        raise SystemExit(
            f"Invalid BUNDLE_PAIRS step {step_raw!r}; expected integer hours."
        )
    key = f"{date_raw}:{int(step_raw)}"
    if key in seen:
        continue
    seen.add(key)
    normalized.append(key)
print(",".join(normalized))
PY
}

require_choice() {
  local value="$1"; shift
  local ok=1
  for c in "$@"; do
    if [[ "${value}" == "${c}" ]]; then
      ok=0
      break
    fi
  done
  if [[ "${ok}" -ne 0 ]]; then
    die "Invalid value '${value}'. Allowed: $*"
  fi
}

require_choice "${STACK_FLAVOR}" new old
require_choice "${SOURCE_HPC}" ac ag leonardo jupiter
[[ "${CHECKPOINT_SHORT}" != REPLACE_* ]] || die "Set CHECKPOINT_SHORT."
[[ -d "${SOURCE_GRIB_ROOT}" ]] || die "SOURCE_GRIB_ROOT does not exist: ${SOURCE_GRIB_ROOT}"
[[ -n "${RUN_ROOT}" ]] || die "RUN_ROOT must not be empty."
[[ -n "${BUNDLE_DIR}" ]] || die "BUNDLE_DIR must not be empty."
[[ -n "${DATES}" ]] || die "DATES must not be empty."
[[ -n "${STEPS}" ]] || die "STEPS must not be empty."
[[ -n "${MEMBERS}" ]] || die "MEMBERS must not be empty."

BUNDLE_PAIRS="$(normalize_bundle_pairs "${BUNDLE_PAIRS}")"

HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}" in
  ac*) HOST_FAMILY="ac" ;;
  ag*) HOST_FAMILY="ag" ;;
  *) die "Unsupported login node family (${HOST_SHORT}). Run from ac-* or ag-*." ;;
esac

REPO_ROOT="/etc/ecmwf/nfs/dh2_home_a/ecm5702/dev/downscaling-tools"

if [[ "${STACK_FLAVOR}" == "new" ]]; then
  CODE_ROOT="${REPO_ROOT}"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-dyn/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag/bin/activate"
  fi
else
  CODE_ROOT="/home/ecm5702/dev/anemoi_2024/downscaling-tools-OLD"
  if [[ "${HOST_FAMILY}" == "ac" ]]; then
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-old/bin/activate"
  else
    VENV_ACTIVATE="/home/ecm5702/dev/.ds-ag-old/bin/activate"
  fi
fi

[[ -d "${CODE_ROOT}" ]] || die "Code root not found: ${CODE_ROOT}"
[[ -f "${VENV_ACTIVATE}" ]] || die "Env activate script not found: ${VENV_ACTIVATE}"

mkdir -p "${RUN_ROOT}/logs" "${BUNDLE_DIR}"

source "${VENV_ACTIVATE}"
BASE_PYTHONPATH="${PYTHONPATH:-}"
if [[ -n "${BASE_PYTHONPATH}" ]]; then
  export PYTHONPATH="${CODE_ROOT}:${BASE_PYTHONPATH}"
else
  export PYTHONPATH="${CODE_ROOT}"
fi
export DATA_DIR=/home/mlx/ai-ml/datasets/
export DATA_STABLE_DIR=/home/mlx/ai-ml/datasets/stable/
export OUTPUT=/ec/res4/scratch/ecm5702/aifs
export GRID_DIR=/home/mlx/ai-ml/grids/
export INTER_MAT_DIR=/home/ecm5702/hpcperm/data/inter_mat
export RESIDUAL_STATISTICS_DIR=/home/ecm5702/hpcperm/data/residuals_statistics/
export BUNDLE_MODULE_DIR=/home/ecm5702/dev/experiments/2026_02_16_clean_grib_prediction

require_source_gribs() {
  local date="$1"
  local required=(
    "${SOURCE_GRIB_ROOT}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_sfc.grib"
    "${SOURCE_GRIB_ROOT}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_pl.grib"
    "${SOURCE_GRIB_ROOT}/enfo_o96_0001_date${date}_time0000_step24to120_sfc.grib"
    "${SOURCE_GRIB_ROOT}/iekm_o96_iekm_date${date}_time0000_step24to120_sfc_y.grib"
    "${SOURCE_GRIB_ROOT}/iekm_o96_iekm_date${date}_time0000_step24to120_pl_y.grib"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || die "Missing required source GRIB: ${path}"
  done
}

unique_dates="$(
  python - "${DATES}" "${BUNDLE_PAIRS}" <<'PY'
import sys

dates_raw = sys.argv[1]
bundle_pairs_raw = sys.argv[2]
seen = []
if bundle_pairs_raw:
    for token in bundle_pairs_raw.split(","):
        token = token.strip()
        if not token:
            continue
        date_raw, _ = token.split(":", 1)
        if date_raw not in seen:
            seen.append(date_raw)
else:
    for token in dates_raw.split(","):
        token = token.strip()
        if token and token not in seen:
            seen.append(token)
print("\n".join(seen))
PY
)"

while IFS= read -r date; do
  [[ -n "${date}" ]] || continue
  require_source_gribs "${date}"
done <<< "${unique_dates}"

build_one_bundle() {
  local date="$1"
  local step="$2"
  local member="$3"
  local bundle_nc
  local -a hres_static_arg=()
  local -a lres_sfc_extra_args=()
  local -a target_sfc_extra_args=()
  local lres_tp_path
  local target_tp_path
  bundle_nc="$(printf "%s/enfo_o48_0001_date%s_time0000_mem%02d_step%03dh_input_bundle.nc" "${BUNDLE_DIR}" "${date}" "${member}" "${step}")"
  if [[ -n "${HRES_STATIC_GRIB}" ]]; then
    hres_static_arg=(--hres-static-grib "${HRES_STATIC_GRIB}")
  fi
  lres_tp_path="${SOURCE_GRIB_ROOT}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_sfc_tp.grib"
  target_tp_path="${SOURCE_GRIB_ROOT}/iekm_o96_iekm_date${date}_time0000_step24to120_sfc_y_tp.grib"
  if [[ -f "${lres_tp_path}" ]]; then
    lres_sfc_extra_args=(--lres-sfc-extra-grib "${lres_tp_path}")
  fi
  if [[ -f "${target_tp_path}" ]]; then
    target_sfc_extra_args=(--target-sfc-extra-grib "${target_tp_path}")
  fi
  python -m manual_inference.prediction.predict build-bundle     --lres-sfc-grib "${SOURCE_GRIB_ROOT}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_sfc.grib"     "${lres_sfc_extra_args[@]}"     --lres-pl-grib "${SOURCE_GRIB_ROOT}/enfo_o48_0001_date${date}_time0000_mem1to10_step24to120_pl.grib"     --hres-grib "${SOURCE_GRIB_ROOT}/enfo_o96_0001_date${date}_time0000_step24to120_sfc.grib"     "${hres_static_arg[@]}"     --target-sfc-grib "${SOURCE_GRIB_ROOT}/iekm_o96_iekm_date${date}_time0000_step24to120_sfc_y.grib"     "${target_sfc_extra_args[@]}"     --target-pl-grib "${SOURCE_GRIB_ROOT}/iekm_o96_iekm_date${date}_time0000_step24to120_pl_y.grib"     --step-hours "${step}"     --member "${member}"     --out "${bundle_nc}"
}

IFS=',' read -r -a MEMBER_LIST <<< "${MEMBERS}"

if [[ -n "${BUNDLE_PAIRS}" ]]; then
  IFS=',' read -r -a PAIR_LIST <<< "${BUNDLE_PAIRS}"
  for pair in "${PAIR_LIST[@]}"; do
    pair="${pair//[[:space:]]/}"
    [[ -n "${pair}" ]] || continue
    date="${pair%%:*}"
    step="${pair##*:}"
    for member in "${MEMBER_LIST[@]}"; do
      member="${member//[[:space:]]/}"
      [[ -n "${member}" ]] || continue
      build_one_bundle "${date}" "${step}" "${member}"
    done
  done
else
  IFS=',' read -r -a DATE_LIST <<< "${DATES}"
  IFS=',' read -r -a STEP_LIST <<< "${STEPS}"
  for date in "${DATE_LIST[@]}"; do
    date="${date//[[:space:]]/}"
    [[ -n "${date}" ]] || continue
    for step in "${STEP_LIST[@]}"; do
      step="${step//[[:space:]]/}"
      [[ -n "${step}" ]] || continue
      for member in "${MEMBER_LIST[@]}"; do
        member="${member//[[:space:]]/}"
        [[ -n "${member}" ]] || continue
        build_one_bundle "${date}" "${step}" "${member}"
      done
    done
  done
fi

python - "${BUNDLE_DIR}" "${DATES}" "${STEPS}" "${BUNDLE_PAIRS}" "${MEMBERS}" "${RUN_ROOT}/bundle_build_verification.json" <<'PY'
import json
import sys
from pathlib import Path

import xarray as xr


def parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


bundle_dir = Path(sys.argv[1])
dates = parse_csv(sys.argv[2])
steps = [int(token) for token in parse_csv(sys.argv[3])]
bundle_pairs_raw = parse_csv(sys.argv[4])
members = [int(token) for token in parse_csv(sys.argv[5])]
out_path = Path(sys.argv[6])

expected: list[Path] = []
if bundle_pairs_raw:
    for token in bundle_pairs_raw:
        date_raw, step_raw = token.split(":", 1)
        for member in members:
            expected.append(
                bundle_dir / (
                    f"enfo_o48_0001_date{date_raw}_time0000_mem{member:02d}_step{int(step_raw):03d}h_input_bundle.nc"
                )
            )
else:
    for date_raw in dates:
        for step in steps:
            for member in members:
                expected.append(
                    bundle_dir / (
                        f"enfo_o48_0001_date{date_raw}_time0000_mem{member:02d}_step{step:03d}h_input_bundle.nc"
                    )
                )

missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise SystemExit(f"Missing rebuilt bundles: {missing[:5]}")

sample = expected[0]
with xr.open_dataset(sample) as ds:
    target_vars = sorted(name for name in ds.variables if name.startswith("target_hres_"))
    if not target_vars:
        raise SystemExit(f"{sample} is missing target_hres_* variables")
    marker = str(ds.attrs.get("has_target_hres_fields", "")).strip().lower()
    if marker not in {"yes", "true", "1"}:
        raise SystemExit(f"{sample} has invalid has_target_hres_fields marker: {marker!r}")

payload = {
    "bundle_dir": str(bundle_dir),
    "expected_bundle_count": len(expected),
    "sample_bundle": str(sample),
    "sample_target_vars": target_vars[:10],
    "truth_marker": "yes",
}
out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

cat <<EOF
[bundle-build] checkpoint_short=${CHECKPOINT_SHORT}
[bundle-build] host=${HOST_SHORT}
[bundle-build] source_grib_root=${SOURCE_GRIB_ROOT}
[bundle-build] bundle_dir=${BUNDLE_DIR}
[bundle-build] dates=${DATES}
[bundle-build] steps=${STEPS}
[bundle-build] bundle_pairs=${BUNDLE_PAIRS:-none}
[bundle-build] members=${MEMBERS}
EOF
