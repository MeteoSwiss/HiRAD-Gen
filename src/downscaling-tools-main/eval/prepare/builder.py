"""Truth-aware bundle builder.

Reads the lane config ``prepare:`` section, which specifies per-lane GRIB
filename templates, and calls ``manual_inference.prediction.predict build-bundle``
for each (date, step, member) combination.

Lane config schema (under ``prepare:``)::

    bundle_filename_tpl: "prefix_date{date}_mem{member:02d}_step{step:03d}h_input_bundle.nc"
    args:
      lres_sfc_grib: "{source_grib_root}/prefix_date{date}_sfc.grib"
      lres_pl_grib:  "{source_grib_root}/prefix_date{date}_pl.grib"
      hres_grib:     "{source_grib_root}/prefix_hres_date{date}_sfc.grib"
      target_sfc_grib: "{source_grib_root}/prefix_target_date{date}_sfc.grib"
      # optional channel overrides (used by o1280_o2560)
      lres_sfc_channels: "10u,10v,2t,msl"
      lres_pl_channels:  "NONE"
    optional_sidecars:
      - file: "{source_grib_root}/prefix_date{date}_tp.grib"
        arg:  lres_sfc_extra_grib

Template variables available in all string values:
  {date}               YYYYMMDD
  {step}               integer forecast step (hours)
  {member}             integer member index
  {source_grib_root}   from --source-grib-root CLI arg
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def build_bundles(
    lane_config: dict,
    bundle_dir: Path,
    source_grib_root: str,
    *,
    dates: list[str] | None = None,
    steps: list[int] | None = None,
    members: list[int] | None = None,
    bundle_pairs: list[str] | None = None,
    verification_path: Path | None = None,
) -> Path:
    """Build truth-aware bundles for all (date, step, member) combinations.

    Resumable: skips bundles whose output NC already exists.
    Verifies all expected bundles after building and writes a verification JSON.
    Returns ``bundle_dir``.
    """
    prepare_cfg = lane_config.get("prepare")
    if not prepare_cfg:
        raise ValueError("Lane config has no 'prepare:' section.")

    predict_cfg = lane_config.get("predict", {})
    dates = dates or predict_cfg.get("dates", [])
    steps = steps or [int(s) for s in predict_cfg.get("steps", [])]
    members = members or [int(m) for m in predict_cfg.get("members", [])]
    bundle_pairs = bundle_pairs or []

    bundle_dir.mkdir(parents=True, exist_ok=True)

    ctx_base = {
        "source_grib_root": source_grib_root,
    }

    filename_tpl: str = prepare_cfg["bundle_filename_tpl"]
    fixed_args: dict[str, str] = prepare_cfg.get("args", {})
    sidecars: list[dict] = prepare_cfg.get("optional_sidecars", [])

    combos = _expand_combos(dates, steps, members, bundle_pairs)
    LOG.info("bundle-build: %d bundle(s) → %s", len(combos), bundle_dir)

    for i, (date, step, member) in enumerate(combos, 1):
        ctx = {**ctx_base, "date": date, "step": step, "member": member}
        bundle_out = bundle_dir / filename_tpl.format(**ctx)

        if bundle_out.exists():
            LOG.info("[%d/%d] skip (exists): %s", i, len(combos), bundle_out.name)
            continue

        LOG.info("[%d/%d] building: %s", i, len(combos), bundle_out.name)
        cmd = [sys.executable, "-m", "manual_inference.prediction.predict", "build-bundle"]

        for arg_name, arg_val in fixed_args.items():
            cmd += [f"--{arg_name.replace('_', '-')}", arg_val.format(**ctx)]
        for sidecar in sidecars:
            sidecar_path = Path(sidecar["file"].format(**ctx))
            if sidecar_path.exists():
                cmd += [f"--{sidecar['arg'].replace('_', '-')}", str(sidecar_path)]
        cmd += ["--step-hours", str(step), "--member", str(member), "--out", str(bundle_out)]
        subprocess.run(cmd, check=True)

    verify_path = verification_path or bundle_dir.parent / "bundle_build_verification.json"
    _verify(
        bundle_dir=bundle_dir,
        filename_tpl=filename_tpl,
        combos=combos,
        ctx_base=ctx_base,
        out_path=verify_path,
    )
    return bundle_dir


def _expand_combos(
    dates: list[str],
    steps: list[int],
    members: list[int],
    bundle_pairs: list[str],
) -> list[tuple[str, int, int]]:
    if bundle_pairs:
        combos: list[tuple[str, int, int]] = []
        for token in bundle_pairs:
            date, step_s = token.split(":")
            for member in members:
                combos.append((date.strip(), int(step_s), member))
        return combos
    return [
        (date, step, member)
        for date in dates
        for step in steps
        for member in members
    ]


def _verify(
    *,
    bundle_dir: Path,
    filename_tpl: str,
    combos: list[tuple[str, int, int]],
    ctx_base: dict,
    out_path: Path,
) -> None:
    import xarray as xr

    missing: list[str] = []
    sample_target_vars: list[str] = []

    for date, step, member in combos:
        ctx = {**ctx_base, "date": date, "step": step, "member": member}
        path = bundle_dir / filename_tpl.format(**ctx)
        if not path.exists():
            missing.append(str(path))
            continue
        if not sample_target_vars:
            with xr.open_dataset(path) as ds:
                target_vars = sorted(v for v in ds.variables if v.startswith("target_hres_"))
                if not target_vars:
                    raise RuntimeError(f"Bundle missing target_hres_* variables: {path}")
                marker = str(ds.attrs.get("has_target_hres_fields", "")).strip().lower()
                if marker not in {"yes", "true", "1"}:
                    raise RuntimeError(
                        f"Bundle has invalid has_target_hres_fields={marker!r}: {path}"
                    )
                sample_target_vars = target_vars[:10]

    if missing:
        raise RuntimeError(f"Missing {len(missing)} bundle(s) after build: {missing[:5]}")

    payload: dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "expected_bundle_count": len(combos),
        "sample_target_vars": sample_target_vars,
        "truth_marker": "yes",
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOG.info("Bundle verification passed (%d bundles): %s", len(combos), out_path)
