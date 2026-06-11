"""ECMWF spectra evaluator — PDF plots."""
from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def plot(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    """Build consolidated spectra PDF from ampl_*.npy files produced by the runner."""
    from ._plotter import build_pdf

    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    amp_dir = results_dir / "spectra"
    if not amp_dir.exists():
        LOG.warning("spectra_ecmwf plotter: spectra dir not found: %s", amp_dir)
        return output_dir

    out_pdf = output_dir / "spectra_ecmwf.pdf"
    try:
        n = build_pdf(amp_dir, out_pdf)
        LOG.info("spectra_ecmwf: wrote %d-page PDF: %s", n, out_pdf)
    except FileNotFoundError as exc:
        LOG.warning("spectra_ecmwf plotter: %s", exc)

    return output_dir
