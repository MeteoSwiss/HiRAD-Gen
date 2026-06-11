"""spectra_plot_pdf.py — Build a consolidated PDF of spectral curves.

Two modes are auto-detected:
  proxy   : spectra_curve_summary.json exists in spectra_dir
  ecmwf   : ampl_*.npy / wvn_*.npy files exist in subdirectories of spectra_dir

Usage:
    python spectra_plot_pdf.py --spectra-dir <dir> --out-pdf <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402


# Preferred variable order for proxy mode
_VAR_ORDER = ["10u", "10v", "2t", "msl", "sp", "t_850", "z_500"]
_SCOPE_ORDER = ["full_field", "residual"]


# ---------------------------------------------------------------------------
# Proxy mode
# ---------------------------------------------------------------------------

def build_pdf_proxy(spectra_dir: Path, out_pdf: Path) -> int:
    """Build PDF from spectra_curve_summary.json.  Returns page count."""
    summary_path = spectra_dir / "spectra_curve_summary.json"
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    run_label = summary.get("run_label", "")
    score_wvn_min = summary.get("score_wavenumber_min_exclusive", None)
    weather_states: dict = summary.get("weather_states", {})

    # Sort variables: known order first, then remaining alphabetically
    known = [v for v in _VAR_ORDER if v in weather_states]
    extras = sorted(v for v in weather_states if v not in _VAR_ORDER)
    var_order = known + extras

    pages = 0
    with PdfPages(out_pdf) as pdf:
        for var in var_order:
            vs = weather_states[var]
            if vs.get("status") != "ok":
                print(f"[WARN] Skipping variable '{var}': status={vs.get('status')}")
                continue
            scopes = vs.get("scopes", {})
            for scope in _SCOPE_ORDER:
                if scope not in scopes:
                    continue
                sc = scopes[scope]
                if sc.get("status") != "ok":
                    print(f"[WARN] Skipping {var}/{scope}: status={sc.get('status')}")
                    continue

                wvn = np.asarray(sc["wavenumbers"], dtype=float)
                pred = np.asarray(sc["prediction_mean"], dtype=float)
                truth = np.asarray(sc["truth_mean"], dtype=float)
                rl2 = sc.get("relative_l2_mean_curve", float("nan"))
                n_curves = sc.get("n_curves", "?")

                # Fix 4: guard against empty mask
                mask = wvn > 0
                if mask.sum() == 0:
                    print(f"[WARN] Skipping {var}/{scope}: no wavenumbers > 0")
                    continue

                # Fix 3: guard against all-NaN or all-zero data after masking
                pred_masked = pred[mask]
                truth_masked = truth[mask]
                if (
                    len(pred_masked) == 0
                    or len(truth_masked) == 0
                    or (np.all(np.isnan(pred_masked)) or np.all(pred_masked == 0))
                    or (np.all(np.isnan(truth_masked)) or np.all(truth_masked == 0))
                ):
                    print(f"[WARN] Skipping {var}/{scope}: no valid data after masking")
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))
                # Only plot positive wavenumbers for log-log
                ax.loglog(wvn[mask], pred_masked, label="prediction", color="tab:blue")
                ax.loglog(wvn[mask], truth_masked, label="truth", color="tab:orange", linestyle="--")

                if score_wvn_min is not None:
                    ax.axvline(score_wvn_min, color="gray", linestyle=":", linewidth=0.8, label=f"score ell>{score_wvn_min:.0f}")

                ax.set_xlabel("Wavenumber ℓ")
                ax.set_ylabel("Spectral amplitude")
                ax.set_title(f"{run_label}  |  {var}  |  {scope}  (n={n_curves})")
                ax.legend(fontsize=8)
                rl2_label = "RL2=N/A" if (isinstance(rl2, float) and np.isnan(rl2)) else f"RL2={rl2:.4f}"
                ax.text(
                    0.98, 0.98,
                    rl2_label,
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.7),
                )
                ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1

    return pages


# ---------------------------------------------------------------------------
# ECMWF npy mode
# ---------------------------------------------------------------------------

def build_pdf_ecmwf(spectra_dir: Path, out_pdf: Path) -> int:
    """Build PDF from ampl_*.npy / wvn_*.npy files in param subdirs.

    Each subdirectory that contains ampl_*.npy files becomes one page.
    Returns page count.
    """
    # Find param dirs: subdirs that contain at least one ampl_*.npy file
    param_dirs = sorted(
        d for d in spectra_dir.iterdir()
        if d.is_dir() and list(d.glob("ampl_*.npy"))
    )

    pages = 0
    with PdfPages(out_pdf) as pdf:
        for param_dir in param_dirs:
            ampl_files = sorted(param_dir.glob("ampl_*.npy"))
            wvn_files = sorted(param_dir.glob("wvn_*.npy"))

            # Fix 1: guard against wvn/ampl count mismatch
            if wvn_files and len(ampl_files) != len(wvn_files):
                print(
                    f"[WARN] Skipping {param_dir.name}: "
                    f"ampl file count ({len(ampl_files)}) != wvn file count ({len(wvn_files)})"
                )
                continue

            # Load all amplitude arrays; use matching wvn if available
            ampls = [np.load(f) for f in ampl_files]
            if wvn_files:
                wvns = [np.load(f) for f in wvn_files]
                wvn = np.mean(np.stack(wvns, axis=0), axis=0)
            else:
                wvn = np.arange(len(ampls[0]), dtype=float)

            # Fix 2: guard against inconsistent amplitude array lengths
            ampl_lengths = [len(a) for a in ampls]
            if len(set(ampl_lengths)) > 1:
                print(f"[WARN] Skipping {param_dir.name}: inconsistent amplitude array lengths")
                continue

            ampl_stack = np.stack(ampls, axis=0)
            ampl_mean = np.mean(ampl_stack, axis=0)
            ampl_std = np.std(ampl_stack, axis=0)

            fig, ax = plt.subplots(figsize=(8, 5))
            mask = wvn > 0
            ax.loglog(wvn[mask], ampl_mean[mask], label="mean", color="tab:blue")
            ax.fill_between(
                wvn[mask],
                np.maximum(ampl_mean[mask] - ampl_std[mask], 1e-30),
                ampl_mean[mask] + ampl_std[mask],
                alpha=0.25,
                color="tab:blue",
                label="±1σ",
            )
            ax.set_xlabel("Wavenumber ℓ")
            ax.set_ylabel("Spectral amplitude")
            ax.set_title(f"{param_dir.name}  (n={len(ampl_files)} files)")
            ax.legend(fontsize=8)
            ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

    return pages


# ---------------------------------------------------------------------------
# Top-level auto-detect
# ---------------------------------------------------------------------------

def build_pdf(spectra_dir: Path | str, out_pdf: Path | str) -> int:
    """Auto-detect mode and build consolidated PDF.

    Returns page count.
    Raises FileNotFoundError if neither proxy summary nor npy files are found.
    """
    spectra_dir = Path(spectra_dir)
    out_pdf = Path(out_pdf)

    summary_path = spectra_dir / "spectra_curve_summary.json"
    if summary_path.exists():
        return build_pdf_proxy(spectra_dir, out_pdf)

    # Check for ECMWF npy subdirs
    has_npy = any(
        list(d.glob("ampl_*.npy"))
        for d in spectra_dir.iterdir()
        if d.is_dir()
    ) if spectra_dir.exists() else False

    if has_npy:
        return build_pdf_ecmwf(spectra_dir, out_pdf)

    raise FileNotFoundError(
        f"No spectra data found in {spectra_dir}: "
        "expected spectra_curve_summary.json (proxy mode) or "
        "ampl_*.npy files in subdirectories (ECMWF mode)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a consolidated spectra PDF from proxy summary or ECMWF npy files."
    )
    parser.add_argument("--spectra-dir", required=True, type=Path, help="Directory with spectra data.")
    parser.add_argument("--out-pdf", required=True, type=Path, help="Output PDF path.")
    args = parser.parse_args()

    n = build_pdf(args.spectra_dir, args.out_pdf)
    print(f"Wrote consolidated PDF ({n} pages): {args.out_pdf}")


if __name__ == "__main__":
    main()
