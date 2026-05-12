"""Tests for eval.jobs.resources."""

from __future__ import annotations

from eval.jobs.resources import resolve_resources, validate_resource_profile


def _host_config():
    return {
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        }
    }


def test_resolve_resources_host_defaults():
    resources = resolve_resources({}, _host_config(), stage="predict")
    assert resources["qos"] == "nf"
    assert resources["time"] == "04:00:00"
    assert resources["mem"] == "64G"
    assert resources["cpus"] == 16
    assert resources["gpus"] == 0
    assert resources["ntasks_per_node"] == 1


def test_resolve_resources_stage_overlay():
    lane = {
        "resource_profiles": {
            "predict": {"qos": "ng", "gpus": 1, "time": "12:00:00", "mem": "0", "cpus": 16}
        }
    }
    resources = resolve_resources(lane, _host_config(), stage="predict")
    assert resources["gpus"] == 1
    assert resources["qos"] == "ng"  # forced by gpus > 0
    assert resources["time"] == "12:00:00"
    assert resources["mem"] == "0"


def test_resolve_resources_evaluator_overlay():
    lane = {
        "resource_profiles": {
            "evaluate": {"time": "04:00:00", "mem": "128G"},
            "evaluate_tc": {"time": "06:00:00", "mem": "256G", "cpus": 4},
        }
    }
    resources = resolve_resources(lane, _host_config(), stage="evaluate", evaluator="tc")
    assert resources["time"] == "06:00:00"
    assert resources["mem"] == "256G"
    assert resources["cpus"] == 4
    assert resources["qos"] == "nf"  # gpus=0 → host default


def test_resolve_resources_gpu_forces_ng():
    lane = {
        "resource_profiles": {
            "evaluate_sigma": {"gpus": 1, "time": "01:00:00"}
        }
    }
    resources = resolve_resources(lane, _host_config(), stage="evaluate", evaluator="sigma")
    assert resources["gpus"] == 1
    assert resources["qos"] == "ng"


def test_resolve_resources_cpu_uses_host_qos():
    lane = {"resource_profiles": {"evaluate": {"cpus": 8}}}
    resources = resolve_resources(lane, _host_config(), stage="evaluate")
    assert resources["gpus"] == 0
    assert resources["qos"] == "nf"


def test_validate_resource_profile_valid():
    errors = validate_resource_profile({
        "qos": "ng", "time": "04:00:00", "mem": "64G", "cpus": 16, "gpus": 1,
    })
    assert errors == []


def test_validate_resource_profile_bad_qos():
    errors = validate_resource_profile({"qos": "invalid"})
    assert len(errors) == 1
    assert "qos" in errors[0]


def test_validate_resource_profile_bad_time():
    errors = validate_resource_profile({"time": "4hours"})
    assert len(errors) == 1
    assert "time" in errors[0]
