#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import metview as mv
import numpy as np


FILE_RE = re.compile(r".*_(?P<date>\d{8})_(?P<step>\d{2,3})_(?P<member>\d+)_nopoles\.grb_sh$")


@dataclass(frozen=True)
class SpectraConfig:
    weather_state: str
    param: str
    level: str
    dir_name: str


CONFIGS: dict[str, SpectraConfig] = {
    "2t": SpectraConfig(weather_state="2t", param="2t", level="sfc", dir_name="2t_sfc"),
    "10u": SpectraConfig(weather_state="10u", param="10u", level="sfc", dir_name="10u_sfc"),
    "10v": SpectraConfig(weather_state="10v", param="10v", level="sfc", dir_name="10v_sfc"),
    "sp": SpectraConfig(weather_state="sp", param="sp", level="sfc", dir_name="sp_sfc"),
    "msl": SpectraConfig(weather_state="msl", param="msl", level="sfc", dir_name="msl_sfc"),
    "t_850": SpectraConfig(weather_state="t_850", param="t", level="850", dir_name="t_850"),
    "z_500": SpectraConfig(weather_state="z_500", param="z", level="500", dir_name="z_500"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectra amplitudes from actual spectral_harmonics outputs."
    )
    parser.add_argument("--spectral-harmonics-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weather-states", default="10u,10v,2t,sp,t_850,z_500")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def parse_weather_states(raw: str) -> list[str]:
    states = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [state for state in states if state not in CONFIGS]
    if unknown:
        raise ValueError(f"Unsupported weather states: {unknown}")
    return states


def parse_components(path: Path) -> tuple[int, int, int]:
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized spectral harmonic filename: {path}")
    return (
        int(match.group("date")),
        int(match.group("step")),
        int(match.group("member")),
    )


def read_curve(path: Path, cfg: SpectraConfig) -> tuple[np.ndarray, np.ndarray]:
    fs_in = mv.Fieldset()
    if cfg.level == "sfc":
        fs_in.append(mv.read(data=mv.read(str(path)), param=cfg.param))
    else:
        fs_in.append(mv.read(data=mv.read(str(path)), levelist=cfg.level, param=cfg.param))
    if len(fs_in) != 1:
        raise RuntimeError(f"Expected exactly one field in {path}, got {len(fs_in)}")
    sp = mv.spec_graph(
        data=fs_in,
        truncation=319,
        x_axis_type="logartihmic",
        y_axis_type="logartihmic",
    )
    wvn = np.array(sp[1]["INPUT_X_VALUES"])
    ampl = np.array(sp[1]["INPUT_Y_VALUES"])
    return wvn, ampl


def main() -> None:
    args = parse_args()
    sh_root = Path(args.spectral_harmonics_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    states = parse_weather_states(args.weather_states)

    written = []
    for state in states:
        cfg = CONFIGS[state]
        in_dir = sh_root / cfg.dir_name
        out_dir = out_root / cfg.dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(in_dir.glob("*_nopoles.grb_sh")):
            date_ymd, step_hours, member = parse_components(path)
            wvn, ampl = read_curve(path, cfg)
            stem = f"{date_ymd}_{step_hours}_{cfg.weather_state}_n{member}"
            wvn_path = out_dir / f"wvn_{stem}.npy"
            ampl_path = out_dir / f"ampl_{stem}.npy"
            np.save(wvn_path, wvn)
            np.save(ampl_path, ampl)
            written.append(
                {
                    "weather_state": cfg.weather_state,
                    "input": str(path),
                    "wavenumbers": str(wvn_path),
                    "amplitudes": str(ampl_path),
                    "date": date_ymd,
                    "step_hours": step_hours,
                    "member": member,
                }
            )

    if not written:
        raise RuntimeError(f"No spectra amplitudes were written from {sh_root}")

    summary = {
        "spectral_harmonics_dir": str(sh_root),
        "out_dir": str(out_root),
        "weather_states": states,
        "written_count": len(written),
        "files": written,
    }
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else (out_root / "spectra_summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote spectra summary: {summary_path}")


if __name__ == "__main__":
    main()
