from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Mapping

import earthkit.data as ekd

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PREPML_O96_SFC_PARAMS = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw"]
PREPML_O96_PL_PARAMS = ["q", "t", "u", "v", "w", "z"]
PREPML_O96_PL_LEVELS = [50, 100, 200, 300, 400, 500, 700, 850, 925, 1000]
PREPML_O320_STATIC_PARAMS = ["z", "lsm"]


def _format_mars_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("MARS request list values cannot be empty")
        return "/".join(str(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        raise TypeError(f"Unsupported nested mapping in MARS request value: {value!r}")
    return str(value)


def mars_request_to_grib(
    request: Mapping[str, Any],
    *,
    target: str | Path,
    workdir: str | Path | None = None,
    cleanup: bool = True,
) -> Path:
    workdir = Path(workdir) if workdir else Path.cwd()
    workdir.mkdir(parents=True, exist_ok=True)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    lines = ["RETRIEVE,"]
    for k, v in request.items():
        lines.append(f"    {k.upper():<10} = {_format_mars_value(v)},")
    lines.append(f"    TARGET     = \"{target}\"")
    payload = "\n".join(lines)

    with tempfile.NamedTemporaryFile("w", dir=workdir, suffix=".mars", delete=False) as f:
        f.write(payload)
        mars_path = Path(f.name)

    LOG.info("Running MARS request via %s", mars_path)
    subprocess.run(["mars", str(mars_path)], check=True)

    if cleanup:
        mars_path.unlink(missing_ok=True)

    return target


def mars_request_to_xarray(
    request: Mapping[str, Any],
    *,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = ekd.from_source("mars", dict(request)).to_xarray()
    ds.to_netcdf(out_path)
    LOG.info("Saved MARS request to %s", out_path)
    return out_path


def _parse_json_arg(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON: {exc}") from exc


def _canonical_date(date_value: str) -> str:
    raw = str(date_value).strip()
    if len(raw) == 8 and raw.isdigit():
        return raw
    try:
        return datetime.fromisoformat(raw).strftime("%Y%m%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid --date value: {date_value!r}") from exc


def _build_default_request(
    *,
    kind: str,
    date: str,
    time: str,
    step: int,
    member: int,
    expver: str,
) -> dict[str, Any]:
    if kind == "o96-sfc":
        return {
            "class": "od",
            "stream": "eefo",
            "type": "pf",
            "expver": expver,
            "grid": "O96",
            "levtype": "sfc",
            "param": PREPML_O96_SFC_PARAMS,
            "date": date,
            "time": time,
            "step": int(step),
            "number": int(member),
        }
    if kind == "o96-pl":
        return {
            "class": "od",
            "stream": "eefo",
            "type": "pf",
            "expver": expver,
            "grid": "O96",
            "levtype": "pl",
            "param": PREPML_O96_PL_PARAMS,
            "levelist": PREPML_O96_PL_LEVELS,
            "date": date,
            "time": time,
            "step": int(step),
            "number": int(member),
        }
    if kind == "o320-static":
        return {
            "class": "od",
            "stream": "enfo",
            "type": "pf",
            "expver": expver,
            "grid": "O320",
            "levtype": "sfc",
            "param": PREPML_O320_STATIC_PARAMS,
            "date": date,
            "time": time,
            "step": int(step),
            "number": int(member),
        }
    raise SystemExit(f"Unsupported --kind value: {kind}")


def _resolve_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_json:
        return _parse_json_arg(args.request_json)
    if not args.kind:
        raise SystemExit("Provide --request-json or --kind for prepml default requests")
    if not args.date:
        raise SystemExit("Provide --date when using --kind defaults")
    return _build_default_request(
        kind=args.kind,
        date=_canonical_date(args.date),
        time=str(args.time),
        step=int(args.step),
        member=int(args.member),
        expver=str(args.expver),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MARS utilities (prefer xarray output).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_grib = sub.add_parser("to-grib", help="Run a MARS request and save GRIB.")
    p_grib.add_argument("--request-json", default="", help="JSON dict of MARS keys.")
    p_grib.add_argument(
        "--kind",
        choices=["o96-sfc", "o96-pl", "o320-static"],
        default="",
        help="Prepml-compatible default request kind (used when --request-json is omitted).",
    )
    p_grib.add_argument("--date", default="", help="Date (YYYYMMDD or ISO date) for default requests.")
    p_grib.add_argument("--time", default="0000", help="Time for default requests (default: 0000).")
    p_grib.add_argument("--step", type=int, default=24, help="Step hours for default requests (default: 24).")
    p_grib.add_argument(
        "--member",
        type=int,
        default=1,
        help="Ensemble member number for default requests (default: 1).",
    )
    p_grib.add_argument("--expver", default="0001", help="expver for default requests (default: 0001).")
    p_grib.add_argument("--target", required=False)
    p_grib.add_argument(
        "--out-root",
        default="/home/ecm5702/hpcperm/data/input_data",
        help="Base output folder for input data.",
    )
    p_grib.add_argument("--grid", default="", help="Resolution tag (e.g. o320, o96).")

    p_xr = sub.add_parser("to-xarray", help="Run a MARS request and save NetCDF.")
    p_xr.add_argument("--request-json", default="", help="JSON dict of MARS keys.")
    p_xr.add_argument(
        "--kind",
        choices=["o96-sfc", "o96-pl", "o320-static"],
        default="",
        help="Prepml-compatible default request kind (used when --request-json is omitted).",
    )
    p_xr.add_argument("--date", default="", help="Date (YYYYMMDD or ISO date) for default requests.")
    p_xr.add_argument("--time", default="0000", help="Time for default requests (default: 0000).")
    p_xr.add_argument("--step", type=int, default=24, help="Step hours for default requests (default: 24).")
    p_xr.add_argument(
        "--member",
        type=int,
        default=1,
        help="Ensemble member number for default requests (default: 1).",
    )
    p_xr.add_argument("--expver", default="0001", help="expver for default requests (default: 0001).")
    p_xr.add_argument("--out", required=False)
    p_xr.add_argument(
        "--out-root",
        default="/home/ecm5702/hpcperm/data/input_data",
        help="Base output folder for input data.",
    )
    p_xr.add_argument("--grid", default="", help="Resolution tag (e.g. o320, o96).")

    args = parser.parse_args()

    if args.cmd == "to-grib":
        request = _resolve_request(args)
        target = args.target
        if not target:
            stream = str(request.get("stream", "mars")).lower()
            grid = args.grid or str(request.get("grid", "grid")).lower()
            date = str(request.get("date", "date"))
            out_dir = Path(args.out_root) / grid
            target = out_dir / f"{stream}_{grid}_{date}.grib"
        mars_request_to_grib(request, target=target)
    elif args.cmd == "to-xarray":
        request = _resolve_request(args)
        out_path = args.out
        if not out_path:
            stream = str(request.get("stream", "mars")).lower()
            grid = args.grid or str(request.get("grid", "grid")).lower()
            date = str(request.get("date", "date"))
            out_dir = Path(args.out_root) / grid
            out_path = out_dir / f"{stream}_{grid}_{date}.nc"
        mars_request_to_xarray(request, out_path=out_path)


if __name__ == "__main__":
    main()
