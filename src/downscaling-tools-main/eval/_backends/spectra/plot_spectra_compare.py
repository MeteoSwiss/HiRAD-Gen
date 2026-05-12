from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.paths import reference_spectra_dir


PARAM_CONFIGS = [
    {"param": "2t", "level": "sfc", "dir_name": "2t_sfc"},
    {"param": "10u", "level": "sfc", "dir_name": "10u_sfc"},
    {"param": "10v", "level": "sfc", "dir_name": "10v_sfc"},
    {"param": "sp", "level": "sfc", "dir_name": "sp_sfc"},
    {"param": "t", "level": "850", "dir_name": "t_850"},
    {"param": "z", "level": "500", "dir_name": "z_500"},
]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _mean_curve(curves: list[np.ndarray]) -> np.ndarray:
    if not curves:
        raise ValueError("Cannot average an empty list of curves.")
    min_len = min(len(curve) for curve in curves)
    trimmed = [curve[:min_len] for curve in curves]
    return np.mean(trimmed, axis=0)


def _build_exp_configs(
    expver: str,
    hres_reference_name: str,
    hres_reference_label: str,
    eefo_token: str,
    hres_token: str,
) -> list[dict]:
    return [
        {
            "name": expver,
            "type": "ai",
            "base_path": f"/home/ecm5702/perm/ai_spectra/{expver}/spectra",
            "label": expver,
        },
        {
            "name": "eefo_o96",
            "type": "hpc",
            "base_path": str(reference_spectra_dir("eefo_o96")),
            "label": "eefo O96",
            "token": eefo_token,
        },
        {
            "name": hres_reference_name,
            "type": "hpc",
            "base_path": str(reference_spectra_dir(hres_reference_name)),
            "label": hres_reference_label,
            "token": hres_token,
        },
    ]


def _get_paths(conf: dict, dir_name: str, date_in: int, step: int, param: str, level: str, number: int) -> tuple[Path, Path]:
    b = conf["base_path"]
    if conf["type"] == "ai":
        w = Path(f"{b}/{dir_name}/wvn_{date_in}_{step}_{param}_{level}_{conf['name']}_n{number}.npy")
        a = Path(f"{b}/{dir_name}/ampl_{date_in}_{step}_{param}_{level}_{conf['name']}_n{number}.npy")
    else:
        token = conf.get("token", "1")
        w = Path(f"{b}/{dir_name}/wvn_{date_in}_{step}_{param}_{level}_{token}_n{number}.npy")
        a = Path(f"{b}/{dir_name}/ampl_{date_in}_{step}_{param}_{level}_{token}_n{number}.npy")
    return w, a


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot spectra comparison: expver vs eefo_o96 vs selected ENFO reference.")
    parser.add_argument("--expver", required=True)
    parser.add_argument("--date-start", default="2023-08-26 00:00:00")
    parser.add_argument("--date-end", default="2023-08-26 00:00:00")
    parser.add_argument("--date-freq", default="1D")
    parser.add_argument("--steps", default="144", help="Comma-separated forecast steps.")
    parser.add_argument("--members", default="1", help="Comma-separated ensemble members.")
    parser.add_argument(
        "--separate-steps",
        action="store_true",
        help="Plot one line per case-step instead of aggregating all selected steps into one line per case.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory for PDFs.")
    parser.add_argument(
        "--hres-reference-name",
        default="enfo_o320",
        help="Reference spectra folder name registered under /home/ecm5702/perm/reference.",
    )
    parser.add_argument(
        "--hres-reference-label",
        default="enfo O320",
        help="Legend label for selected high-resolution reference.",
    )
    parser.add_argument(
        "--eefo-token",
        default="1",
        help="Filename token for eefo_o96 spectra files (default: 1).",
    )
    parser.add_argument(
        "--hres-token",
        default="1",
        help="Filename token for selected high-resolution reference spectra files (default: 1).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"/home/ecm5702/scratch/eval/{args.expver}")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_list = pd.date_range(pd.Timestamp(args.date_start), pd.Timestamp(args.date_end), freq=args.date_freq)
    steps = _parse_csv_ints(args.steps)
    members = _parse_csv_ints(args.members)
    expvers = _build_exp_configs(
        args.expver,
        hres_reference_name=args.hres_reference_name,
        hres_reference_label=args.hres_reference_label,
        eefo_token=args.eefo_token,
        hres_token=args.hres_token,
    )
    colors = [cmc.batlow(i / max(1, len(expvers) - 1)) for i in range(len(expvers))]
    type_style = {"ai": "-", "hpc": "--"}

    for cfg in PARAM_CONFIGS:
        param, level, dir_name = cfg["param"], cfg["level"], cfg["dir_name"]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        any_data = False
        for ie, conf in enumerate(expvers):
            miss = 0
            found = 0
            step_w_curves: list[np.ndarray] = []
            step_a_curves: list[np.ndarray] = []
            for step in steps:
                member_w, member_a = [], []
                for number in members:
                    all_w, all_a = [], []
                    for d in date_list:
                        date_in = d.year * 10000 + d.month * 100 + d.day
                        wfp, afp = _get_paths(conf, dir_name, date_in, step, param, level, number)
                        if wfp.exists() and afp.exists():
                            all_w.append(np.load(wfp))
                            all_a.append(np.load(afp))
                            found += 1
                        else:
                            miss += 1
                    if all_w and all_a:
                        member_w.append(np.stack(all_w, axis=1))
                        member_a.append(np.stack(all_a, axis=1))
                if member_w and member_a:
                    avg_w = np.mean([arr.mean(axis=1) for arr in member_w], axis=0)
                    avg_a = np.mean([arr.mean(axis=1) for arr in member_a], axis=0)
                    step_w_curves.append(avg_w)
                    step_a_curves.append(avg_a)
                    if args.separate_steps:
                        iok = range(3, len(avg_w))
                        ax.plot(
                            avg_w[iok],
                            avg_a[iok],
                            color=colors[ie],
                            linestyle=type_style.get(conf["type"], "-"),
                            linewidth=2.3,
                            label=f"{conf['label']} step={step}",
                        )
                        any_data = True
            if miss and not found:
                warnings.warn(f"No spectra found for {conf['name']} param={param} level={level}")
            if step_w_curves and step_a_curves and not args.separate_steps:
                avg_w = _mean_curve(step_w_curves)
                avg_a = _mean_curve(step_a_curves)
                iok = range(3, len(avg_w))
                ax.plot(
                    avg_w[iok],
                    avg_a[iok],
                    color=colors[ie],
                    linestyle=type_style.get(conf["type"], "-"),
                    linewidth=2.3,
                    label=conf["label"],
                )
                any_data = True

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Zonal wavenumber")
        ax.set_ylabel("Mean power")
        ax.set_title(f"{param} @ {level}")
        ax.grid(color="grey", linestyle="--", linewidth=0.25)
        ax.legend(loc="lower left", frameon=False, fontsize=8)
        if any_data:
            out = output_dir / f"spectra_{param}_{level}.pdf"
            plt.tight_layout()
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
