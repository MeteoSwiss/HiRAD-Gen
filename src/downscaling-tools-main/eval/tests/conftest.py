"""Shared fixtures for eval framework tests."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path


def pytest_configure(config):
    config.addinivalue_line("markers", "hpc: marks tests requiring HPC resources")


@pytest.fixture
def valid_lane_dict():
    """Minimal valid lane config dict."""
    return {
        "predict": {
            "members": [1, 2],
            "steps": [24, 48],
            "dates": ["20230826", "20230827"],
            "sampler": {"schedule_type": "karras", "num_steps": 25},
        },
        "evaluator_groups": {
            "default": ["tc", "spectra", "surface", "region_plot"],
        },
    }


@pytest.fixture
def valid_host_dict():
    """Minimal valid host config dict."""
    return {
        "code_root": "/home/ecm5702/dev/downscaling-tools",
        "scratch_root": "/home/ecm5702/scratch",
        "scheduler": {
            "qos": "nf",
            "default_time": "04:00:00",
            "default_mem": "64G",
            "default_cpus": 16,
        },
        "environment_setup": {
            "module_loads": ["ecmwf-toolbox"],
            "venv_activate": "/home/ecm5702/dev/.ds-dyn/bin/activate",
        },
    }


@pytest.fixture
def valid_event_dict():
    """Minimal valid event config dict."""
    return {
        "name": "test_event",
        "lat_min": 10.0,
        "lat_max": 40.0,
        "lon_min": -100.0,
        "lon_max": -80.0,
        "dates": ["26", "27", "28"],
    }


@pytest.fixture
def config_dir(tmp_path, valid_lane_dict, valid_host_dict, valid_event_dict):
    """Temp config directory with valid lane/host/event YAML files."""
    for sub in ("lanes", "hosts", "events"):
        (tmp_path / sub).mkdir()
    (tmp_path / "lanes" / "test_lane.yaml").write_text(yaml.dump(valid_lane_dict))
    (tmp_path / "hosts" / "test_host.yaml").write_text(yaml.dump(valid_host_dict))
    (tmp_path / "events" / "test_event.yaml").write_text(yaml.dump(valid_event_dict))
    return tmp_path


@pytest.fixture
def predictions_dir(tmp_path):
    """Tmp directory with canonical and non-matching prediction files."""
    pdir = tmp_path / "predictions"
    pdir.mkdir()
    for date in ["20230826", "20230827"]:
        for step in [24, 48, 72]:
            (pdir / f"predictions_{date}_step{step:03d}.nc").touch()
    (pdir / "random_file.txt").touch()
    (pdir / "predictions_bad_name.nc").touch()
    return pdir
