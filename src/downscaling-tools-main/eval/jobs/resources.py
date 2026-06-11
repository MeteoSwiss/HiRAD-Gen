"""Resource profile resolution for the evaluation pipeline.

Resolves per-stage SLURM resource requirements by merging host scheduler
defaults with lane-level ``resource_profiles`` overrides.
"""

from __future__ import annotations

import re
from typing import Any


_VALID_QOS = {"nf", "ng"}
_TIME_RE = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")
_MEM_RE = re.compile(r"^(\d+[GMKT]|0)$")


def validate_resource_profile(profile: dict[str, Any]) -> list[str]:
    """Validate a single resource profile dict.  Returns list of error strings."""
    errors: list[str] = []
    if "qos" in profile and profile["qos"] not in _VALID_QOS:
        errors.append(f"qos must be one of {sorted(_VALID_QOS)}, got {profile['qos']!r}")
    if "time" in profile and not _TIME_RE.match(str(profile["time"])):
        errors.append(f"time must match HH:MM:SS, got {profile['time']!r}")
    if "mem" in profile and not _MEM_RE.match(str(profile["mem"])):
        errors.append(f"mem must match \\d+[GMKT] or '0', got {profile['mem']!r}")
    if "cpus" in profile:
        if not isinstance(profile["cpus"], int) or profile["cpus"] < 1:
            errors.append(f"cpus must be int > 0, got {profile['cpus']!r}")
    if "gpus" in profile:
        if not isinstance(profile["gpus"], int) or profile["gpus"] < 0:
            errors.append(f"gpus must be int >= 0, got {profile['gpus']!r}")
    if "ntasks_per_node" in profile:
        if not isinstance(profile["ntasks_per_node"], int) or profile["ntasks_per_node"] < 1:
            errors.append(f"ntasks_per_node must be int > 0, got {profile['ntasks_per_node']!r}")
    return errors


def resolve_resources(
    lane_config: dict,
    host_config: dict,
    stage: str,
    evaluator: str | None = None,
) -> dict[str, Any]:
    """Resolve SLURM resource requirements for a pipeline stage.

    Parameters
    ----------
    lane_config : dict
        Lane configuration (from ``load_lane``).
    host_config : dict
        Host configuration (from ``load_host``).
    stage : str
        Pipeline stage: ``"predict"``, ``"evaluate"``, or ``"scoreboard"``.
    evaluator : str or None
        Evaluator name (e.g. ``"tc"``, ``"spectra"``).  Only used when
        *stage* is ``"evaluate"`` to look up evaluator-specific overrides.

    Returns
    -------
    dict
        Resource dict with keys: qos, time, mem, cpus, gpus, ntasks_per_node.
    """
    scheduler = host_config["scheduler"]

    # 1. Start with host scheduler defaults
    resources: dict[str, Any] = {
        "qos": scheduler["qos"],
        "time": scheduler.get("default_time", "04:00:00"),
        "mem": scheduler.get("default_mem", "64G"),
        "cpus": scheduler.get("default_cpus", 16),
        "gpus": 0,
        "ntasks_per_node": 1,
    }

    profiles = lane_config.get("resource_profiles", {})

    # 2. Overlay stage-level profile
    if stage in profiles:
        resources.update(profiles[stage])

    # 3. Overlay evaluator-specific profile
    if evaluator and f"evaluate_{evaluator}" in profiles:
        resources.update(profiles[f"evaluate_{evaluator}"])

    # 4. Host QOS fixup (last step)
    if resources["gpus"] == 0:
        resources["qos"] = scheduler["qos"]
    else:
        resources["qos"] = "ng"

    return resources
