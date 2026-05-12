from __future__ import annotations

import pytest

from eval._backends.region_plotting.plotting.config import (
    DEFAULT_MODEL_VARIABLES,
    DEFAULT_WEATHER_STATES,
    GRID_CONFIG,
    PREDICTION_REGION_BOXES,
)


@pytest.fixture
def prediction_region_boxes() -> dict[str, list[float]]:
    return PREDICTION_REGION_BOXES


def test_prediction_region_boxes_is_non_empty_dict(prediction_region_boxes: dict[str, list[float]]) -> None:
    assert isinstance(prediction_region_boxes, dict)
    assert prediction_region_boxes


@pytest.mark.parametrize("region_name", ["amazon_forest", "andes_central", "eastern_us_coast"])
def test_known_prediction_regions_exist(
    prediction_region_boxes: dict[str, list[float]], region_name: str
) -> None:
    assert region_name in prediction_region_boxes


def test_all_region_boxes_have_four_valid_bounds(prediction_region_boxes: dict[str, list[float]]) -> None:
    for region_name, bounds in prediction_region_boxes.items():
        assert len(bounds) == 4, region_name
        lat_min, lat_max, lon_min, lon_max = bounds
        assert lat_min < lat_max, region_name
        assert lon_min < lon_max, region_name
        assert -90.0 <= lat_min <= 90.0, region_name
        assert -90.0 <= lat_max <= 90.0, region_name
        assert -360.0 <= lon_min <= 360.0, region_name
        assert -360.0 <= lon_max <= 360.0, region_name


def test_default_model_variables_is_non_empty() -> None:
    assert isinstance(DEFAULT_MODEL_VARIABLES, list)
    assert DEFAULT_MODEL_VARIABLES


def test_default_weather_states_is_non_empty() -> None:
    assert isinstance(DEFAULT_WEATHER_STATES, list)
    assert DEFAULT_WEATHER_STATES


@pytest.mark.parametrize("grid_name", ["O96", "O1280", "O2560"])
def test_grid_config_contains_expected_entries(grid_name: str) -> None:
    assert grid_name in GRID_CONFIG
    assert isinstance(GRID_CONFIG[grid_name], dict)
    assert GRID_CONFIG[grid_name]
