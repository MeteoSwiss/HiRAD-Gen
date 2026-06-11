"""Configuration loader with validation for the evaluation framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent

_LANE_REQUIRED_KEYS = {"predict", "evaluator_groups"}
_LANE_ALLOWED_KEYS = {
    "predict", "tc", "spectra", "surface", "regions",
    "evaluator_groups", "sigma", "mechanistic", "intermediate",
    "resource_profiles", "region_plot", "prepare",
}

_HOST_REQUIRED_KEYS = {"code_root", "scratch_root", "scheduler", "environment_setup"}

_EVENT_REQUIRED_KEYS = {"name", "lat_min", "lat_max", "lon_min", "lon_max", "dates"}


class ConfigValidationError(Exception):
    pass


def _deep_merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigValidationError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _validate_lane(config: dict[str, Any], path: Path) -> None:
    for key in _LANE_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required top-level key '{key}'")

    unknown = set(config.keys()) - _LANE_ALLOWED_KEYS
    if unknown:
        raise ConfigValidationError(
            f"{path}: unknown top-level key(s) {sorted(unknown)} "
            f"(allowed: {', '.join(sorted(_LANE_ALLOWED_KEYS))})"
        )

    predict = config["predict"]
    if not isinstance(predict, dict):
        raise ConfigValidationError(f"{path}: 'predict' must be a mapping, got {type(predict).__name__}")

    if "members" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.members' (expected list of int)")
    if not isinstance(predict["members"], list) or not all(isinstance(m, int) for m in predict["members"]):
        raise ConfigValidationError(
            f"{path}: 'predict.members' must be a list of int, got {predict['members']!r}"
        )

    if "steps" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.steps' (expected list of int)")
    if not isinstance(predict["steps"], list) or not all(isinstance(s, int) for s in predict["steps"]):
        raise ConfigValidationError(
            f"{path}: 'predict.steps' must be a list of int, got {predict['steps']!r}"
        )

    if "dates" not in predict:
        raise ConfigValidationError(f"{path}: missing required key 'predict.dates' (expected list of str)")
    if not isinstance(predict["dates"], list) or not all(isinstance(d, str) for d in predict["dates"]):
        raise ConfigValidationError(
            f"{path}: 'predict.dates' must be a list of str, got {predict['dates']!r}"
        )

    evaluator_groups = config["evaluator_groups"]
    if not isinstance(evaluator_groups, dict):
        raise ConfigValidationError(
            f"{path}: 'evaluator_groups' must be a mapping, got {type(evaluator_groups).__name__}"
        )
    if "default" not in evaluator_groups:
        raise ConfigValidationError(f"{path}: missing required key 'evaluator_groups.default' (expected list of str)")
    if not isinstance(evaluator_groups["default"], list):
        raise ConfigValidationError(
            f"{path}: 'evaluator_groups.default' must be a list of str, got {evaluator_groups['default']!r}"
        )


def _validate_host(config: dict[str, Any], path: Path) -> None:
    for key in _HOST_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required key '{key}'")

    code_root = config["code_root"]
    if not isinstance(code_root, str) or not code_root.startswith("/"):
        raise ConfigValidationError(
            f"{path}: key 'code_root' must be an absolute path (starts with '/'), got {code_root!r}"
        )

    scratch_root = config["scratch_root"]
    if not isinstance(scratch_root, str) or not scratch_root.startswith("/"):
        raise ConfigValidationError(
            f"{path}: key 'scratch_root' must be an absolute path (starts with '/'), got {scratch_root!r}"
        )

    scheduler = config["scheduler"]
    if not isinstance(scheduler, dict):
        raise ConfigValidationError(f"{path}: 'scheduler' must be a mapping, got {type(scheduler).__name__}")
    if "qos" not in scheduler or not isinstance(scheduler["qos"], str):
        raise ConfigValidationError(f"{path}: 'scheduler.qos' must be a str")
    if "default_time" not in scheduler or not isinstance(scheduler["default_time"], str):
        raise ConfigValidationError(f"{path}: 'scheduler.default_time' must be a str")

    env = config["environment_setup"]
    if not isinstance(env, dict):
        raise ConfigValidationError(f"{path}: 'environment_setup' must be a mapping, got {type(env).__name__}")
    if "module_loads" not in env or not isinstance(env["module_loads"], list):
        raise ConfigValidationError(f"{path}: 'environment_setup.module_loads' must be a list")
    if "venv_activate" not in env or not isinstance(env["venv_activate"], str):
        raise ConfigValidationError(f"{path}: 'environment_setup.venv_activate' must be a str")


def _validate_event(config: dict[str, Any], path: Path) -> None:
    for key in _EVENT_REQUIRED_KEYS:
        if key not in config:
            raise ConfigValidationError(f"{path}: missing required key '{key}'")

    if not isinstance(config["name"], str):
        raise ConfigValidationError(f"{path}: 'name' must be a str, got {type(config['name']).__name__}")

    for coord in ("lat_min", "lat_max", "lon_min", "lon_max"):
        val = config[coord]
        if not isinstance(val, (int, float)):
            raise ConfigValidationError(f"{path}: '{coord}' must be numeric, got {type(val).__name__}")

    if not isinstance(config["dates"], list):
        raise ConfigValidationError(f"{path}: 'dates' must be a list, got {type(config['dates']).__name__}")


def load_lane(name: str, overrides: dict | None = None) -> dict:
    path = _CONFIG_DIR / "lanes" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Lane config not found: {path}")
    config = _load_yaml(path)
    if overrides:
        config = _deep_merge(config, overrides)
    _validate_lane(config, path)
    return config


def load_host(name: str, overrides: dict | None = None) -> dict:
    path = _CONFIG_DIR / "hosts" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Host config not found: {path}")
    config = _load_yaml(path)
    if overrides:
        config = _deep_merge(config, overrides)
    _validate_host(config, path)
    return config


def load_event(name: str) -> dict:
    path = _CONFIG_DIR / "events" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Event config not found: {path}")
    config = _load_yaml(path)
    _validate_event(config, path)
    return config
