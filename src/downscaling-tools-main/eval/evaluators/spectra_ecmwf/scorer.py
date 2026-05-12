"""ECMWF spectra evaluator — scoring.

The ECMWF route produces ampl_*.npy / wvn_*.npy files (one per param subdir).
Scoring against a reference baseline requires matched reference npy files,
which are not yet provisioned. Returns empty until reference data is set up.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def score(
    results_dir: str | Path,
    lane_config: dict,
    eval_config: dict,
    **kwargs,
) -> list[dict[str, Any]]:
    """Score ECMWF spectra results.

    Returns empty list until reference baseline npy files are configured
    under eval_config['reference_dir'].
    """
    results_dir = Path(results_dir)
    reference_dir = eval_config.get("reference_dir")
    if not reference_dir:
        LOG.info(
            "spectra_ecmwf scorer: no reference_dir configured; skipping scoring. "
            "Set eval_config['reference_dir'] to enable."
        )
        return []

    # Placeholder: implement L2 scoring against reference npy files when
    # reference_dir is provisioned.
    LOG.warning("spectra_ecmwf scorer: reference_dir set but scoring not yet implemented.")
    return []
