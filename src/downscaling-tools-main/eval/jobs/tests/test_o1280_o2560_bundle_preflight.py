from __future__ import annotations

import xarray as xr

from eval.jobs import o1280_o2560_bundle_preflight as preflight


def _surface_ds(*names: str) -> xr.Dataset:
    return xr.Dataset(data_vars={name: ("points", [1.0, 2.0]) for name in names})


def test_build_preflight_result_accepts_cfgrib_surface_names(monkeypatch):
    datasets = {
        ("lres_input.grib", "surface"): _surface_ds("u10", "v10", "t2m", "msl"),
        ("lres_input.grib", "isobaricInhPa"): _surface_ds(),
        ("forcing_sfc.grib", None): _surface_ds("z", "lsm"),
        ("target_y.grib", None): _surface_ds("u10", "v10", "t2m", "msl"),
    }

    def _fake_open(path, *, filter_by_keys=None):
        level = None if filter_by_keys is None else filter_by_keys.get("typeOfLevel")
        return datasets[(str(path), level)], None

    monkeypatch.setattr(preflight, "_maybe_open_dataset", _fake_open)
    monkeypatch.setattr(preflight, "_select_scope", lambda ds, **_: ds)

    result = preflight.build_preflight_result(
        lres_input_grib="lres_input.grib",
        hres_forcing_grib="forcing_sfc.grib",
        target_grib="target_y.grib",
        step_hours=120,
        member=1,
    )

    assert result["strict_bundle_ready"] is True
    assert result["detected_level_splits"]["lres_surface"]["variables"] == ["10u", "10v", "2t", "msl"]
    assert result["detected_level_splits"]["target_surface"]["variables"] == ["10u", "10v", "2t", "msl"]


def test_build_preflight_result_reports_strict_ready(monkeypatch):
    datasets = {
        ("lres_input.grib", "surface"): _surface_ds("10u", "10v", "2t", "msl"),
        ("lres_input.grib", "isobaricInhPa"): _surface_ds(),
        ("forcing_sfc.grib", None): _surface_ds("z", "lsm"),
        ("target_y.grib", None): _surface_ds("10u", "10v", "2t", "msl"),
    }

    def _fake_open(path, *, filter_by_keys=None):
        level = None if filter_by_keys is None else filter_by_keys.get("typeOfLevel")
        return datasets[(str(path), level)], None

    monkeypatch.setattr(preflight, "_maybe_open_dataset", _fake_open)
    monkeypatch.setattr(preflight, "_select_scope", lambda ds, **_: ds)

    result = preflight.build_preflight_result(
        lres_input_grib="lres_input.grib",
        hres_forcing_grib="forcing_sfc.grib",
        target_grib="target_y.grib",
        step_hours=120,
        member=1,
    )

    assert result["strict_bundle_ready"] is True
    assert result["blockers"] == []
    assert result["missing"]["lres_surface"] == []
    assert result["missing"]["hres_static"] == []
    assert result["missing"]["target_surface"] == []
    assert result["contract"]["output_weather_states_csv"] == "10u,10v,2t,msl"


def test_build_preflight_result_reports_missing_surface_contract(monkeypatch):
    datasets = {
        ("lres_input.grib", "surface"): _surface_ds("10u", "10v", "2t"),
        ("lres_input.grib", "isobaricInhPa"): _surface_ds(),
        ("forcing_sfc.grib", None): _surface_ds("z"),
        ("target_y.grib", None): _surface_ds("10u", "10v", "2t"),
    }

    def _fake_open(path, *, filter_by_keys=None):
        level = None if filter_by_keys is None else filter_by_keys.get("typeOfLevel")
        return datasets[(str(path), level)], None

    monkeypatch.setattr(preflight, "_maybe_open_dataset", _fake_open)
    monkeypatch.setattr(preflight, "_select_scope", lambda ds, **_: ds)

    result = preflight.build_preflight_result(
        lres_input_grib="lres_input.grib",
        hres_forcing_grib="forcing_sfc.grib",
        target_grib="target_y.grib",
        step_hours=120,
        member=1,
    )

    assert result["strict_bundle_ready"] is False
    assert result["missing"]["lres_surface"] == ["msl"]
    assert result["missing"]["hres_static"] == ["lsm"]
    assert result["missing"]["target_surface"] == ["msl"]
    assert "Missing low-res surface variables: msl" in result["blocker_summary"]
