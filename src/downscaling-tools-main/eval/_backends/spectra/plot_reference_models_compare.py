import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval.paths import DEFAULT_REFERENCE_SPECTRA_BASE


PARAM_CONFIGS = [
    {"param": "2t", "level": "sfc", "dir_name": "2t_sfc"},
    {"param": "10u", "level": "sfc", "dir_name": "10u_sfc"},
    {"param": "10v", "level": "sfc", "dir_name": "10v_sfc"},
    {"param": "sp", "level": "sfc", "dir_name": "sp_sfc"},
    {"param": "t", "level": "850", "dir_name": "t_850"},
    {"param": "z", "level": "500", "dir_name": "z_500"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare physical-model reference spectra. Uses step=144 when available, else highest non-zero step."
    )
    p.add_argument(
        "--base-dir",
        default=DEFAULT_REFERENCE_SPECTRA_BASE,
        help="Base directory containing reference spectra model aliases.",
    )
    p.add_argument(
        "--models",
        default="eefo_o96,eefo_o320,enfo_o320,enfo_o1280,destine_o2560_i4ql,destine_o2560_i4ql_step72_100d",
        help="Comma-separated model directory names under base-dir.",
    )
    p.add_argument(
        "--output-dir",
        default="/home/ecm5702/scratch/eval/physical_model_spectra_compare_20260303",
        help="Where to save comparison plots and summary JSON.",
    )
    p.add_argument(
        "--prefer-step",
        type=int,
        default=144,
        help="Preferred step when available.",
    )
    return p.parse_args()


def discover_steps(model_dir: Path) -> list[int]:
    steps = set()
    rx = re.compile(r"^wvn_\d{8}_(\d+)_")
    for f in model_dir.rglob("wvn_*.npy"):
        m = rx.match(f.name)
        if m:
            steps.add(int(m.group(1)))
    return sorted(steps)


def choose_step(steps: list[int], prefer_step: int) -> int | None:
    if not steps:
        return None
    if prefer_step in steps:
        return prefer_step
    non_zero = [s for s in steps if s > 0]
    return max(non_zero) if non_zero else None


def collect_arrays(model_dir: Path, param_dir: str, param: str, level: str, step: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    w_arrays: list[np.ndarray] = []
    a_arrays: list[np.ndarray] = []
    pattern = f"wvn_*_{step}_{param}_{level}_*_n*.npy"
    for w_path in sorted((model_dir / param_dir).glob(pattern)):
        a_path = w_path.with_name(w_path.name.replace("wvn_", "ampl_", 1))
        if not a_path.exists():
            continue
        try:
            w = np.load(w_path)
            a = np.load(a_path)
        except Exception:
            continue
        if w.shape != a.shape or w.size < 8:
            continue
        w_arrays.append(w)
        a_arrays.append(a)
    return w_arrays, a_arrays


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    model_info: list[dict] = []
    for name in model_names:
        d = base_dir / name
        steps = discover_steps(d) if d.exists() else []
        chosen = choose_step(steps, args.prefer_step)
        model_info.append(
            {
                "name": name,
                "path": str(d),
                "exists": d.exists(),
                "steps_with_wvn": steps,
                "chosen_step": chosen,
            }
        )

    cmap = plt.get_cmap("tab10")
    line_styles = ["-", "--", "-.", ":"]

    per_param_stats: dict[str, dict] = {}
    for cfg in PARAM_CONFIGS:
        param = cfg["param"]
        level = cfg["level"]
        dir_name = cfg["dir_name"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        plotted = 0
        stats = {}

        for i, m in enumerate(model_info):
            chosen = m["chosen_step"]
            if not m["exists"] or chosen is None:
                stats[m["name"]] = {"status": "no_wvn", "files": 0}
                continue

            w_arrays, a_arrays = collect_arrays(Path(m["path"]), dir_name, param, level, chosen)
            if not w_arrays:
                stats[m["name"]] = {"status": "no_param_data", "step": chosen, "files": 0}
                continue

            avg_w = np.mean(np.stack(w_arrays), axis=0)
            avg_a = np.mean(np.stack(a_arrays), axis=0)
            iok = np.arange(3, len(avg_w))
            ax.plot(
                avg_w[iok],
                avg_a[iok],
                color=cmap(i % 10),
                linestyle=line_styles[i % len(line_styles)],
                linewidth=1.6,
                label=f"{m['name']} (step {chosen})",
            )
            plotted += 1
            stats[m["name"]] = {"status": "plotted", "step": chosen, "files": len(w_arrays)}

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Zonal wavenumber")
        ax.set_ylabel("Mean power")
        ax.set_title(f"{param} at {level}")
        ax.grid(color="grey", linestyle="--", linewidth=0.25, alpha=0.6)
        if plotted:
            ax.legend(loc="best", frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No comparable spectra found", transform=ax.transAxes, ha="center", va="center")
        out = output_dir / f"physical_models_spectra_{param}_{level}.pdf"
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        per_param_stats[f"{param}_{level}"] = stats
        print(f"Saved {out}")

    summary = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "prefer_step": args.prefer_step,
        "models": model_info,
        "per_param": per_param_stats,
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
