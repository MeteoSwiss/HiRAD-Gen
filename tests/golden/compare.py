"""Compare a golden capture against reference captures.

    python -m tests.golden.compare --ref /path/to/refs --new /path/to/new [--setup X] [--verbose]

Both roots contain ``<setup>/<stage>/`` directories written by ``capture.py``. Compared:
records (arrays element-wise, summaries by hash then statistics, JSON by value) and every file
the entry point wrote under ``run/`` (torch files element-wise; GRIB files message by message,
metadata keys exactly, values within tolerance).

Tolerance policy. Everything up to the network input (dataset, normalization, interpolation,
regridding, conditioning assembly, seeds, patch positions) is deterministic and must be
identical. The GroupNorm mode of the capture (``manifest.json``, see ``capture.py``) decides
the rest:

- ``torch``: every record and file must be identical, network outputs included.
- ``apex`` (production): the apex GroupNorm kernels are not bitwise reproducible, so records
  computed from network outputs are compared relative to their own scale with two bounds: a
  loose pointwise bound (worst single value) and a tight aggregate bound (RMS error, per-channel
  means/stds, scalar sums). Run-to-run noise touches a few values slightly and averages out; a
  behaviour change shifts aggregates.

Reference and new capture must use the same mode. Exit code 1 on any difference beyond tolerance.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .recorder import load_records

GRIB_KEYS = (
    "edition", "centre", "shortName", "paramId", "typeOfLevel", "level", "dataDate", "dataTime",
    "stepType", "stepRange", "step", "startStep", "endStep", "gridType", "Ni", "Nj",
    "numberOfValues", "packingType", "bitsPerValue", "bitmapPresent", "missingValue",
)
SKIPPED_RUN_FILES = (".hydra",)
AGGREGATE_STATS = ("channel_mean", "channel_std", "channel_abs_mean")
POINTWISE_STATS = ("channel_min", "channel_max", "sampled_values")
MODEL_DERIVED = (
    # The regression network input is pure data path (exact); the diffusion input carries the
    # regression mean and, in rollout, the previous prediction.
    re.compile(r"(net_output|diffusion/unet_input|outputs/|inputs/prev_hr|loss/[a-z]+_(sum|tensor)|grad/|saved/param_)"),
    re.compile(r"^run/.*-(predictions|regression-prediction)$"),
    re.compile(r"^run/grib-(pred|ensemble-mean|regression-pred)/"),
)


@dataclass(frozen=True)
class Tolerance:
    pointwise: float = 0.0
    aggregate: float = 0.0


EXACT = Tolerance()
# Calibrated on baseline-8bfd985 against an independent re-capture (2026-11-11). Worst run-to-run
# spread: generate pointwise 5.6e-3 (single outlier, next 1.9e-3) / aggregate 5.9e-4 (tp after the
# inverse Box-Cox); train (bf16) pointwise 2.3e-2 / aggregate 3.3e-3. Negative control (sampler
# 17 instead of 18 steps): generate pointwise 3.0e-2 / aggregate 1.1e-2.
DEFAULT_TOLERANCE = {"generate": Tolerance(pointwise=2e-2, aggregate=3e-3),
                     "train": Tolerance(pointwise=2e-1, aggregate=3e-2)}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""


def is_model_derived(name: str) -> bool:
    return any(pattern.search(name) for pattern in MODEL_DERIVED)


def _per_channel(array: np.ndarray) -> np.ndarray:
    """(channels, values) view: axis -3 is the channel axis of (..., C, H, W) fields; anything
    with fewer dimensions (a GRIB message, a vector) is a single channel."""
    if array.ndim < 3:
        return array.reshape(1, -1)
    return np.moveaxis(array, -3, 0).reshape(array.shape[-3], -1)


def _relative_errors(ref64: np.ndarray, new64: np.ndarray) -> tuple[float, float, int]:
    """Worst-channel errors, each channel relative to its own variability so that fields in
    physical units with very different magnitudes (2t in K, tp in m) are equally sensitive:
    max|diff| / range(ref) and RMS(diff) / std(ref)."""
    worst_point, worst_rms, worst_channel = 0.0, 0.0, 0
    for c, (ref_c, new_c) in enumerate(zip(_per_channel(ref64), _per_channel(new64))):
        valid = ~np.isnan(ref_c)
        if not valid.any():
            continue
        ref_v, diff = ref_c[valid], new_c[valid] - ref_c[valid]
        magnitude = float(np.abs(ref_v).max()) or 1.0
        value_range = float(ref_v.max() - ref_v.min()) or magnitude
        std = float(ref_v.std()) or magnitude
        point = float(np.abs(diff).max()) / value_range
        rms = float(np.sqrt(np.mean(diff ** 2))) / std
        if max(point, rms) > max(worst_point, worst_rms):
            worst_channel = c
        worst_point, worst_rms = max(worst_point, point), max(worst_rms, rms)
    return worst_point, worst_rms, worst_channel


def _array_diff(ref: np.ndarray, new: np.ndarray, tol: Tolerance) -> tuple[bool, str]:
    """Worst-channel max error relative to the channel's range within ``tol.pointwise``, RMS error
    relative to the channel's std within ``tol.aggregate`` (identical required when both are 0)."""
    if ref.shape != new.shape:
        return False, f"shape {ref.shape} -> {new.shape}"
    dtype_ok = ref.dtype == new.dtype
    dtype_note = "" if dtype_ok else f"dtype {ref.dtype} -> {new.dtype}; "
    if ref.dtype.kind not in "fc" and new.dtype.kind not in "fc":
        equal = np.array_equal(ref, new)
        return equal and dtype_ok, dtype_note + ("identical" if equal else f"{int((ref != new).sum())} elements differ")
    ref64, new64 = ref.astype(np.float64), new.astype(np.float64)
    ref_nan, new_nan = np.isnan(ref64), np.isnan(new64)
    if not np.array_equal(ref_nan, new_nan):
        return False, dtype_note + f"NaN pattern differs ({int(ref_nan.sum())} -> {int(new_nan.sum())} NaNs)"
    valid = ~ref_nan
    if not valid.any():
        return dtype_ok, dtype_note + "all NaN"
    if np.array_equal(ref64[valid], new64[valid]):
        return dtype_ok, dtype_note + "identical"
    point_rel, rms_rel, channel = _relative_errors(ref64, new64)
    ok = dtype_ok and point_rel <= tol.pointwise and rms_rel <= tol.aggregate
    return ok, dtype_note + f"point_rel={point_rel:.3e} rms_rel={rms_rel:.3e} (worst channel {channel})"


def _values_close(ref, new, rtol: float) -> bool:
    if isinstance(ref, bool) or isinstance(new, bool) or ref is None or new is None:
        return ref == new
    if isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        if isinstance(ref, float) and isinstance(new, float) and math.isnan(ref) and math.isnan(new):
            return True
        return ref == new or math.isclose(ref, new, rel_tol=rtol, abs_tol=0.0)
    if isinstance(ref, dict) and isinstance(new, dict):
        return ref.keys() == new.keys() and all(_values_close(ref[k], new[k], rtol) for k in ref)
    if isinstance(ref, list) and isinstance(new, list):
        return len(ref) == len(new) and all(_values_close(a, b, rtol) for a, b in zip(ref, new))
    return ref == new


def _compare_summary(ref: dict, new: dict, tol: Tolerance) -> tuple[bool, str]:
    """Per-channel statistics relative to that channel's std (aggregates) or range (extremes);
    sampled values relative to the overall range. Constant channels fall back to the overall scale."""
    if ref["sha256"] == new["sha256"]:
        return True, "identical"
    if ref["shape"] != new["shape"] or ref["dtype"] != new["dtype"]:
        return False, f"{ref['shape']} {ref['dtype']} -> {new['shape']} {new['dtype']}"
    ref_min, ref_max = np.array(ref["channel_min"], dtype=np.float64), np.array(ref["channel_max"], dtype=np.float64)
    overall = float(max(np.abs(ref_min).max(), np.abs(ref_max).max())) or 1.0
    channel_range = np.where(ref_max > ref_min, ref_max - ref_min, overall)
    channel_std = np.array(ref["channel_std"], dtype=np.float64)
    channel_std = np.where(channel_std > 0, channel_std, overall)
    scales = {key: channel_std for key in AGGREGATE_STATS}
    scales.update(channel_min=channel_range, channel_max=channel_range,
                  sampled_values=float(ref_max.max() - ref_min.min()) or overall)
    worst = {}
    for group, keys in (("aggregate", AGGREGATE_STATS), ("pointwise", POINTWISE_STATS)):
        rel, where = 0.0, ""
        for key in keys:
            delta = np.abs(np.array(new[key], dtype=np.float64) - np.array(ref[key], dtype=np.float64)) / scales[key]
            i = int(delta.argmax())
            if delta[i] > rel:
                rel, where = float(delta[i]), f"{key}[{i}]"
        worst[group] = (rel, where)
    ok = worst["aggregate"][0] <= tol.aggregate and worst["pointwise"][0] <= tol.pointwise
    detail = (f"hash differs; aggregate rel={worst['aggregate'][0]:.3e} ({worst['aggregate'][1]}), "
              f"pointwise rel={worst['pointwise'][0]:.3e} ({worst['pointwise'][1]})")
    return ok, detail


def _short(value, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _json_detail(ref, new) -> str:
    if ref == new:
        return "identical"
    if isinstance(ref, (int, float)) and isinstance(new, (int, float)) and ref:
        return f"rel={abs(new - ref) / abs(ref):.3e}"
    return f"{_short(ref)} -> {_short(new)}"


def compare_records(ref_dir: Path, new_dir: Path, model_tol: Tolerance) -> list[Result]:
    ref_index, new_index = load_records(ref_dir), load_records(new_dir)
    results = []
    for name, ref in ref_index.items():
        new = new_index.get(name)
        if new is None:
            results.append(Result(name, False, "missing in new capture"))
            continue
        if ref["kind"] != new["kind"]:
            results.append(Result(name, False, f"kind {ref['kind']} -> {new['kind']}"))
            continue
        tol = model_tol if is_model_derived(name) else EXACT
        if ref["kind"] == "array":
            ok, detail = _array_diff(np.load(ref_dir / "records" / ref["file"]),
                                     np.load(new_dir / "records" / new["file"]), tol)
        elif ref["kind"] == "summary":
            ok, detail = _compare_summary(ref["value"], new["value"], tol)
        else:
            # Scalars computed from network outputs (loss sums, gradient norms) are aggregates.
            ok = _values_close(ref["value"], new["value"], tol.aggregate)
            detail = _json_detail(ref["value"], new["value"])
        results.append(Result(name, ok, detail))
    for name in sorted(set(new_index) - set(ref_index)):
        results.append(Result(name, True, "new record (not in reference)"))
    return results


def _run_files(run_dir: Path) -> set[str]:
    return {
        str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
        if p.is_file() and not p.name.endswith(".log")
        and not any(part in SKIPPED_RUN_FILES for part in p.relative_to(run_dir).parts)
    }


def _read_grib(path: Path) -> list[tuple[dict, np.ndarray]]:
    import eccodes

    messages = []
    with open(path, "rb") as f:
        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                meta = {}
                for key in GRIB_KEYS:
                    try:
                        meta[key] = eccodes.codes_get(gid, key)
                    except eccodes.KeyValueNotFoundError:
                        meta[key] = None
                messages.append((meta, eccodes.codes_get_values(gid)))
            finally:
                eccodes.codes_release(gid)
    return messages


def _compare_grib(ref: Path, new: Path, tol: Tolerance) -> tuple[bool, str]:
    ref_msgs, new_msgs = _read_grib(ref), _read_grib(new)
    if len(ref_msgs) != len(new_msgs):
        return False, f"{len(ref_msgs)} -> {len(new_msgs)} messages"
    details = []
    for i, ((ref_meta, ref_vals), (new_meta, new_vals)) in enumerate(zip(ref_msgs, new_msgs)):
        changed = {k: (ref_meta[k], new_meta[k]) for k in GRIB_KEYS if ref_meta[k] != new_meta[k]}
        if changed:
            return False, f"message {i} ({ref_meta['shortName']}): metadata {changed}"
        ok, detail = _array_diff(ref_vals, new_vals, tol)
        if not ok:
            return False, f"message {i} ({ref_meta['shortName']}): {detail}"
        if detail != "identical":
            details.append(f"{ref_meta['shortName']}: {detail}")
    return True, f"{len(ref_msgs)} messages, metadata identical; " + ("; ".join(details) or "values identical")


def _load_torch(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def compare_run_files(ref_dir: Path, new_dir: Path, model_tol: Tolerance) -> list[Result]:
    ref_run, new_run = ref_dir / "run", new_dir / "run"
    if not ref_run.exists():
        return []
    ref_files, new_files = _run_files(ref_run), _run_files(new_run)
    results = [Result(f"run/{f}", False, "missing in new run") for f in sorted(ref_files - new_files)]
    results += [Result(f"run/{f}", False, "unexpected new file") for f in sorted(new_files - ref_files)]
    for rel_path in sorted(ref_files & new_files):
        name = f"run/{rel_path}"
        tol = model_tol if is_model_derived(name) else EXACT
        ref_path, new_path = ref_run / rel_path, new_run / rel_path
        if rel_path.endswith(".grib"):
            ok, detail = _compare_grib(ref_path, new_path, tol)
        elif rel_path.endswith((".json", ".txt", ".yaml")):
            ok = ref_path.read_bytes() == new_path.read_bytes()
            detail = "identical" if ok else "content differs"
        else:
            ok, detail = _array_diff(_load_torch(ref_path), _load_torch(new_path), tol)
        results.append(Result(name, ok, detail))
    return results


def group_norm_mode(stage_dir: Path) -> str:
    """GroupNorm implementation a capture ran with; captures predating the option used apex."""
    with open(stage_dir / "manifest.json") as f:
        return json.load(f).get("group_norm", "apex")


def compare_stage(ref_dir: Path, new_dir: Path, model_tol: Tolerance) -> list[Result]:
    if not (new_dir / "records.json").exists():
        return [Result("capture", False, f"no capture at {new_dir}")]
    ref_mode, new_mode = group_norm_mode(ref_dir), group_norm_mode(new_dir)
    if ref_mode != new_mode:
        return [Result("capture", False, f"GroupNorm mode differs: reference {ref_mode}, new {new_mode}")]
    if ref_mode == "torch":
        model_tol = EXACT
    return compare_records(ref_dir, new_dir, model_tol) + compare_run_files(ref_dir, new_dir, model_tol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--setup", action="append", help="restrict to these setups (repeatable)")
    parser.add_argument("--stage", action="append", help="restrict to these stages (repeatable)")
    for stage, tol in DEFAULT_TOLERANCE.items():
        parser.add_argument(f"--{stage}-pointwise", type=float, default=tol.pointwise,
                            help=f"worst-value relative tolerance for network-derived {stage} records "
                                 "(apex captures only; torch captures are compared exactly)")
        parser.add_argument(f"--{stage}-aggregate", type=float, default=tol.aggregate,
                            help=f"aggregate (RMS / statistics / sums) relative tolerance for {stage} records "
                                 "(apex captures only)")
    parser.add_argument("--verbose", action="store_true", help="also list matching records")
    args = parser.parse_args()
    tolerance = {stage: Tolerance(getattr(args, f"{stage}_pointwise"), getattr(args, f"{stage}_aggregate"))
                 for stage in DEFAULT_TOLERANCE}

    stage_dirs = sorted(p.parent for p in args.ref.glob("*/*/records.json"))
    stage_dirs = [d for d in stage_dirs
                  if (not args.setup or d.parent.name in args.setup) and (not args.stage or d.name in args.stage)]
    if not stage_dirs:
        raise SystemExit(f"no reference captures under {args.ref}")

    n_failed = 0
    for ref_dir in stage_dirs:
        label = f"{ref_dir.parent.name}/{ref_dir.name}"
        model_tol = tolerance["generate" if ref_dir.name == "generate" else "train"]
        results = compare_stage(ref_dir, args.new / ref_dir.parent.name / ref_dir.name, model_tol)
        failed = [r for r in results if not r.ok]
        n_failed += len(failed)
        n_exact = sum(r.detail.startswith("identical") or "values identical" in r.detail for r in results)
        mode = group_norm_mode(ref_dir)
        print(f"{'FAIL' if failed else 'PASS'}  {label} [{mode} GroupNorm, "
              f"{'exact' if mode == 'torch' else 'tolerance'}]: {len(results) - len(failed)}/{len(results)} match "
              f"({n_exact} identical)")
        for r in results if args.verbose else failed:
            print(f"    {'ok  ' if r.ok else 'DIFF'} {r.name}: {r.detail}")
    print(f"\n{len(stage_dirs)} stage(s) compared, {n_failed} difference(s) beyond tolerance")
    raise SystemExit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
