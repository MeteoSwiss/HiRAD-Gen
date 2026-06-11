"""TC experiment configuration per event. All values come from CLI args at runtime."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TCExperimentConfig:
    """Experiment IDs for a TC event.

    CLI --analysis-expid sets analysis_expid at runtime.
    CLI --extra-reference-expids adds references at runtime.
    """

    analysis_expid: str | None = None
    reference_expids: tuple[str, ...] = ()
    base_tc_dir: str | None = None


# Event registry — all values come from CLI args at runtime.
# The flow helpers are the canonical per-lane source of truth.
EXPERIMENT_CONFIGS: dict[str, TCExperimentConfig] = {
    "franklin": TCExperimentConfig(),
    "idalia": TCExperimentConfig(),
    "hilary": TCExperimentConfig(),
    "dora": TCExperimentConfig(),
    "fernanda": TCExperimentConfig(),
    "humberto": TCExperimentConfig(),
}
