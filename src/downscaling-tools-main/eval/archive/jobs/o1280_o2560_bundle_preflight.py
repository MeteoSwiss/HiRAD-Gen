from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.jobs.o1280_o2560_contract import O1280_O2560_DESTINE_CONTRACT
from eval.jobs.o1280_o2560_contract import contract_to_dict
from manual_inference.input_data_construction import bundle as bundle_mod


CFGRIB_TO_SFC = {value: key for key, value in bundle_mod.SFC_TO_CFGRIB.items()}


def _canonicalize_surface_vars(names: list[str]) -> list[str]:
    return sorted(CFGRIB_TO_SFC.get(name, name) for name in names)


def _sorted_dataset_vars(ds) -> list[str]:
    return sorted(str(name) for name in ds.data_vars)


def _maybe_open_dataset(path: str | Path, *, filter_by_keys: dict[str, str] | None = None):
    if not path:
        return None, None
    try:
        ds = bundle_mod._open_cfgrib_dataset(path, filter_by_keys=filter_by_keys)
    except Exception as exc:  # pragma: no cover - exercised via higher-level result assertions
        return None, str(exc)
    return ds, None


def _select_scope(ds, *, step_hours: int | None, member: int | None, allow_missing_member: bool = False):
    if ds is None:
        return None
    if step_hours is not None:
        ds = bundle_mod._select_step(ds, step_hours)
    if member is not None:
        ds = bundle_mod._select_member(ds, member, allow_missing=allow_missing_member)
    return ds


def _missing(required: tuple[str, ...], available: list[str]) -> list[str]:
    available_set = set(available)
    return [name for name in required if name not in available_set]


def build_preflight_result(
    *,
    lres_input_grib: str | Path,
    hres_forcing_grib: str | Path,
    target_grib: str | Path | None = None,
    step_hours: int | None = None,
    member: int | None = None,
) -> dict[str, Any]:
    contract = O1280_O2560_DESTINE_CONTRACT
    contract_payload = contract_to_dict(contract)

    lres_surface, lres_surface_error = _maybe_open_dataset(
        lres_input_grib,
        filter_by_keys=contract.cfgrib_filters["lres_surface"],
    )
    lres_surface = _select_scope(lres_surface, step_hours=step_hours, member=member)
    lres_pl, lres_pl_error = _maybe_open_dataset(
        lres_input_grib,
        filter_by_keys=contract.cfgrib_filters["lres_pressure_levels"],
    )
    lres_pl = _select_scope(lres_pl, step_hours=step_hours, member=member)
    hres_surface, hres_surface_error = _maybe_open_dataset(hres_forcing_grib)
    hres_surface = _select_scope(
        hres_surface,
        step_hours=step_hours,
        member=member,
        allow_missing_member=True,
    )
    target_surface = None
    target_surface_error = None
    if target_grib:
        target_surface, target_surface_error = _maybe_open_dataset(target_grib)
        target_surface = _select_scope(
            target_surface,
            step_hours=step_hours,
            member=member,
            allow_missing_member=True,
        )

    detected = {
        "lres_surface": {
            "path": str(lres_input_grib),
            "filter_by_keys": contract.cfgrib_filters["lres_surface"],
            "variables": (
                _canonicalize_surface_vars(_sorted_dataset_vars(lres_surface))
                if lres_surface is not None
                else []
            ),
            "error": lres_surface_error,
        },
        "lres_pressure_levels": {
            "path": str(lres_input_grib),
            "filter_by_keys": contract.cfgrib_filters["lres_pressure_levels"],
            "variables": _sorted_dataset_vars(lres_pl) if lres_pl is not None else [],
            "error": lres_pl_error,
        },
        "hres_surface": {
            "path": str(hres_forcing_grib),
            "variables": _sorted_dataset_vars(hres_surface) if hres_surface is not None else [],
            "error": hres_surface_error,
        },
        "target_surface": {
            "path": str(target_grib) if target_grib else "",
            "variables": (
                _canonicalize_surface_vars(_sorted_dataset_vars(target_surface))
                if target_surface is not None
                else []
            ),
            "error": target_surface_error,
        },
    }

    missing_lres_surface = _missing(
        contract.lres_sfc_channels,
        detected["lres_surface"]["variables"],
    )
    missing_hres_static = _missing(
        contract.hres_static_channels,
        detected["hres_surface"]["variables"],
    )
    missing_target_surface = (
        _missing(contract.target_sfc_channels, detected["target_surface"]["variables"])
        if target_grib
        else list(contract.target_sfc_channels)
    )

    blockers: list[str] = []
    if lres_surface_error:
        blockers.append(f"Could not open low-res surface view: {lres_surface_error}")
    if hres_surface_error:
        blockers.append(f"Could not open high-res forcing view: {hres_surface_error}")
    if missing_lres_surface:
        blockers.append(
            "Missing low-res surface variables: " + ", ".join(missing_lres_surface)
        )
    if missing_hres_static:
        blockers.append(
            "Missing high-res explicit static variables: " + ", ".join(missing_hres_static)
        )
    if target_grib:
        if target_surface_error:
            blockers.append(f"Could not open target surface view: {target_surface_error}")
        if missing_target_surface:
            blockers.append(
                "Missing target surface variables: " + ", ".join(missing_target_surface)
            )
    else:
        blockers.append("Missing target GRIB for strict truth-aware bundle build.")

    strict_bundle_ready = not blockers
    proof_only_ready = True
    proof_only_reason = (
        "Debug dataloader fallback remains available once checkpoint_profile passes, "
        "even when the strict bundle contract is not satisfied."
    )

    return {
        "lane": contract.lane,
        "contract": contract_payload,
        "inputs": {
            "lres_input_grib": str(lres_input_grib),
            "hres_forcing_grib": str(hres_forcing_grib),
            "target_grib": str(target_grib) if target_grib else "",
            "step_hours": step_hours,
            "member": member,
        },
        "detected_level_splits": detected,
        "missing": {
            "lres_surface": missing_lres_surface,
            "hres_static": missing_hres_static,
            "target_surface": missing_target_surface,
        },
        "strict_bundle_ready": strict_bundle_ready,
        "proof_only_ready": proof_only_ready,
        "proof_only_reason": proof_only_reason,
        "blockers": blockers,
        "blocker_summary": "none" if not blockers else " | ".join(blockers),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect DestinE o1280->o2560 GRIB inputs against the maintained "
            "surface-only strict-bundle contract and emit machine-readable JSON."
        )
    )
    parser.add_argument("--lres-input-grib", required=True)
    parser.add_argument("--hres-forcing-grib", required=True)
    parser.add_argument("--target-grib", default="")
    parser.add_argument("--step-hours", type=int, default=None)
    parser.add_argument("--member", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_preflight_result(
        lres_input_grib=args.lres_input_grib,
        hres_forcing_grib=args.hres_forcing_grib,
        target_grib=args.target_grib or None,
        step_hours=args.step_hours,
        member=args.member,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
