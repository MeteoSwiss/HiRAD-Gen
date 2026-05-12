import sys
import types

import numpy as np
import torch
import pytest
import xarray as xr

from manual_inference.prediction.dataset import build_predictions_dataset
from manual_inference.prediction.dataset import resolve_output_weather_states
from manual_inference.prediction import predict


class _DummyData:
    def __init__(self, x_in, x_in_hres, y, lon_lres, lat_lres, lon_hres, lat_hres, dates):
        self._x_in = x_in
        self._x_in_hres = x_in_hres
        self._y = y
        self.longitudes = [lon_lres, None, lon_hres]
        self.latitudes = [lat_lres, None, lat_hres]
        self.dates = dates

    def __getitem__(self, idx):
        return self._x_in[idx], self._x_in_hres[idx], self._y[idx]


class _DummyDataModule:
    def __init__(self, data, name_to_idx_in, name_to_idx_out):
        class _DataIndices:
            def __init__(self, name_to_idx_in, name_to_idx_out):
                class _Input:
                    def __init__(self, name_to_idx):
                        self.name_to_index = name_to_idx

                class _Model:
                    def __init__(self, name_to_idx):
                        self.output = _Input(name_to_idx)

                self.data = type("_Data", (), {"input": [_Input(name_to_idx_in)]})
                self.model = _Model(name_to_idx_out)

        self.ds_valid = type("_DS", (), {"data": data})
        self.data_indices = _DataIndices(name_to_idx_in, name_to_idx_out)


class _DummyModel:
    def __init__(self, grid_hres, n_states):
        self.grid_hres = grid_hres
        self.n_states = n_states
        self.calls = []

    def predict_step(self, x_l, x_h, extra_args=None, model_comm_group=None):
        self.calls.append(
            {
                "extra_args": extra_args,
                "x_l_shape": tuple(x_l.shape),
                "x_h_shape": tuple(x_h.shape),
            }
        )
        shape = (1, 1, 1, self.grid_hres, self.n_states)
        return torch.ones(shape, dtype=torch.float32)

    def interpolate_down(self, x, grad_checkpoint=False):
        mean_state = x.mean(dim=1, keepdim=True)
        return mean_state.repeat(1, self.grid_hres, 1)


def test_predict_from_dataloader_shapes_and_members():
    n_samples = 2
    n_vars_in = 3
    n_vars_out = 2
    n_ens = 2
    grid_lres = 4
    grid_hres = 5

    x_in = np.random.rand(n_samples, n_vars_in, n_ens, grid_lres).astype(np.float32)
    x_in_hres = np.random.rand(n_samples, n_vars_in, n_ens, grid_hres).astype(
        np.float32
    )
    y = np.random.rand(n_samples, n_vars_out, n_ens, grid_hres).astype(np.float32)

    lon_lres = np.linspace(0, 3, grid_lres).astype(np.float32)
    lat_lres = np.linspace(10, 13, grid_lres).astype(np.float32)
    lon_hres = np.linspace(0, 4, grid_hres).astype(np.float32)
    lat_hres = np.linspace(20, 24, grid_hres).astype(np.float32)
    dates = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")

    data = _DummyData(
        x_in=x_in,
        x_in_hres=x_in_hres,
        y=y,
        lon_lres=lon_lres,
        lat_lres=lat_lres,
        lon_hres=lon_hres,
        lat_hres=lat_hres,
        dates=dates,
    )

    name_to_idx_in = {"a": 0, "b": 1, "c": 2}
    name_to_idx_out = {"b": 0, "c": 1}
    datamodule = _DummyDataModule(data, name_to_idx_in, name_to_idx_out)
    model = _DummyModel(grid_hres=grid_hres, n_states=len(name_to_idx_out))

    out = predict._predict_from_dataloader(
        inference_model=model,
        datamodule=datamodule,
        device="cpu",
        idx=0,
        n_samples=n_samples,
        members=[0, 1],
        extra_args={},
        precision="fp32",
        model_comm_group=None,
    )

    x_out, y_out, y_pred, lon_l, lat_l, lon_h, lat_h, states, out_dates = out
    assert x_out.shape == (n_samples, 2, grid_lres, len(name_to_idx_out))
    assert y_out.shape == (n_samples, 2, grid_hres, n_vars_out)
    assert y_pred.shape == (n_samples, 2, grid_hres, len(name_to_idx_out))
    assert np.allclose(lon_l, lon_lres)
    assert np.allclose(lat_l, lat_lres)
    assert np.allclose(lon_h, lon_hres)
    assert np.allclose(lat_h, lat_hres)
    assert states == ["b", "c"]
    assert np.all(out_dates == dates)


def test_predict_from_dataloader_no_members():
    x_in = np.zeros((1, 1, 1, 2), dtype=np.float32)
    x_in_hres = np.zeros((1, 1, 1, 3), dtype=np.float32)
    y = np.zeros((1, 1, 1, 3), dtype=np.float32)
    data = _DummyData(
        x_in=x_in,
        x_in_hres=x_in_hres,
        y=y,
        lon_lres=np.zeros(2, dtype=np.float32),
        lat_lres=np.zeros(2, dtype=np.float32),
        lon_hres=np.zeros(3, dtype=np.float32),
        lat_hres=np.zeros(3, dtype=np.float32),
        dates=np.array(["2024-01-01"], dtype="datetime64[D]"),
    )
    datamodule = _DummyDataModule(data, {"a": 0}, {"a": 0})
    model = _DummyModel(grid_hres=3, n_states=1)

    try:
        predict._predict_from_dataloader(
            inference_model=model,
            datamodule=datamodule,
            device="cpu",
            idx=0,
            n_samples=1,
            members=[],
            extra_args={},
            precision="fp32",
            model_comm_group=None,
        )
    except ValueError as exc:
        assert "No members selected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty members")


def test_predict_from_bundle_minimal(monkeypatch):
    point_lres = 3
    point_hres = 4
    monkeypatch.setattr(predict, "open_bundle_dataset", lambda path: xr.Dataset())
    monkeypatch.setattr(predict, "find_missing_explicit_hres_inputs", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        predict,
        "load_inputs_from_bundle_numpy",
        lambda *args, **kwargs: (
            np.stack(
                [
                    np.ones(point_lres, dtype=np.float32),
                    np.full(point_lres, 2.0, dtype=np.float32),
                ],
                axis=1,
            ),
            np.zeros((point_hres, 1), dtype=np.float32),
            np.arange(point_lres, dtype=np.float32),
            np.arange(point_lres, dtype=np.float32) + 10,
            np.arange(point_hres, dtype=np.float32),
            np.arange(point_hres, dtype=np.float32) + 20,
        ),
    )
    monkeypatch.setattr(
        predict,
        "extract_target_from_bundle",
        lambda bundle_nc, weather_states: (None, 0),
    )

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"b": 0, "c": 1}}),
                        type("_In", (), {"name_to_index": {"z": 0}}),
                    ]
                },
            )
            self.model = type("_Model", (), {"output": type("_Out", (), {"name_to_index": {"b": 0, "c": 1}})})

    class _DummyDM:
        def __init__(self):
            self.data_indices = _Indices()
            self.ds_valid = type(
                "_DS",
                (),
                {
                    "data": type(
                        "_Data",
                        (),
                        {
                            "longitudes": [np.arange(point_lres), None, np.arange(point_hres)],
                            "latitudes": [np.arange(point_lres) + 10, None, np.arange(point_hres) + 20],
                        },
                    )
                },
            )

    model = _DummyModel(grid_hres=point_hres, n_states=2)
    dm = _DummyDM()

    out = predict._predict_from_bundle(
        inference_model=model,
        datamodule=dm,
        device="cpu",
        bundle_nc="/tmp/fake.nc",
        member_index=0,
        extra_args={},
        precision="fp32",
        model_comm_group=None,
    )

    x_out, y_out, y_pred, lon_l, lat_l, lon_h, lat_h, states, dates = out
    assert y_out is None
    assert x_out.shape == (1, point_lres, 2)
    assert y_pred.shape == (1, 1, point_hres, 2)
    assert states == ["b", "c"]
    assert dates is None
    assert np.allclose(lon_l, np.arange(point_lres))
    assert np.allclose(lat_h, np.arange(point_hres) + 20)


def test_predict_from_bundle_rejects_missing_explicit_hres_inputs(tmp_path):
    bundle_path = tmp_path / "bundle.nc"
    xr.Dataset(
        data_vars={
            "in_lres_b": ("point_lres", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            "in_hres_z": ("point_hres", np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)),
        },
        coords={
            "point_lres": np.arange(3, dtype=np.int32),
            "point_hres": np.arange(4, dtype=np.int32),
            "lat_lres": ("point_lres", np.array([10.0, 11.0, 12.0], dtype=np.float32)),
            "lon_lres": ("point_lres", np.array([20.0, 21.0, 22.0], dtype=np.float32)),
            "lat_hres": ("point_hres", np.array([30.0, 31.0, 32.0, 33.0], dtype=np.float32)),
            "lon_hres": ("point_hres", np.array([40.0, 41.0, 42.0, 43.0], dtype=np.float32)),
        },
        attrs={"case_valid_time": "2024-01-01T00:00:00"},
    ).to_netcdf(bundle_path)

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"b": 0}}),
                        type("_In", (), {"name_to_index": {"2t": 0, "z": 1}}),
                    ]
                },
            )
            self.model = type(
                "_Model",
                (),
                {"output": type("_Out", (), {"name_to_index": {"2t": 0, "msl": 1}})},
            )

    class _DummyDM:
        def __init__(self):
            self.data_indices = _Indices()

    model = _DummyModel(grid_hres=4, n_states=2)
    dm = _DummyDM()

    with pytest.raises(
        ValueError,
        match="Bundle inference is incompatible with this checkpoint/bundle combination",
    ):
        predict._predict_from_bundle(
            inference_model=model,
            datamodule=dm,
            device="cpu",
            bundle_nc=str(bundle_path),
            member_index=0,
            extra_args={},
            precision="fp32",
            model_comm_group=None,
        )


def test_predict_from_dataloader_forwards_classic_sampling_args():
    x_in = np.random.rand(1, 2, 1, 3).astype(np.float32)
    x_in_hres = np.random.rand(1, 2, 1, 4).astype(np.float32)
    y = np.random.rand(1, 2, 1, 4).astype(np.float32)
    data = _DummyData(
        x_in=x_in,
        x_in_hres=x_in_hres,
        y=y,
        lon_lres=np.arange(3, dtype=np.float32),
        lat_lres=np.arange(3, dtype=np.float32),
        lon_hres=np.arange(4, dtype=np.float32),
        lat_hres=np.arange(4, dtype=np.float32),
        dates=np.array(["2024-01-01"], dtype="datetime64[D]"),
    )
    datamodule = _DummyDataModule(data, {"a": 0, "b": 1}, {"a": 0, "b": 1})
    model = _DummyModel(grid_hres=4, n_states=2)
    classic = {
        "num_steps": 40,
        "sigma_max": 1000.0,
        "sigma_min": 0.03,
        "rho": 7.0,
        "sampler": "heun",
    }

    predict._predict_from_dataloader(
        inference_model=model,
        datamodule=datamodule,
        device="cpu",
        idx=0,
        n_samples=1,
        members=[0],
        extra_args=classic,
        precision="fp32",
        model_comm_group=None,
    )
    assert len(model.calls) == 1
    assert model.calls[0]["extra_args"] == classic


def test_resolve_ckpt_path_falls_back_to_single_ckpt(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run_dir = ckpt_root / "run123"
    run_dir.mkdir(parents=True)
    only_ckpt = run_dir / "anemoi-by_epoch-epoch_021-step_100000.ckpt"
    only_ckpt.write_text("x", encoding="utf-8")

    resolved = predict._resolve_ckpt_path("run123", str(ckpt_root))

    assert resolved == str(only_ckpt)


def test_resolve_ckpt_path_rejects_ambiguous_ckpts(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run_dir = ckpt_root / "run123"
    run_dir.mkdir(parents=True)
    (run_dir / "first.ckpt").write_text("x", encoding="utf-8")
    (run_dir / "second.ckpt").write_text("y", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Multiple base checkpoint files found"):
        predict._resolve_ckpt_path("run123", str(ckpt_root))


def test_resolve_ckpt_path_prefers_base_ckpt_over_inference_companion(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run_dir = ckpt_root / "run123"
    run_dir.mkdir(parents=True)
    base_ckpt = run_dir / "anemoi-by_epoch-epoch_021-step_100000.ckpt"
    inference_ckpt = run_dir / "inference-anemoi-by_epoch-epoch_021-step_100000.ckpt"
    base_ckpt.write_text("base", encoding="utf-8")
    inference_ckpt.write_text("inference", encoding="utf-8")

    resolved = predict._resolve_ckpt_path("run123", str(ckpt_root))

    assert resolved == str(base_ckpt)


def test_resolve_ckpt_path_rejects_inference_companion_input(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run_dir = ckpt_root / "run123"
    run_dir.mkdir(parents=True)
    inference_ckpt = run_dir / "inference-anemoi-by_epoch-epoch_021-step_100000.ckpt"
    inference_ckpt.write_text("inference", encoding="utf-8")

    with pytest.raises(ValueError, match="Pass the base checkpoint path"):
        predict._resolve_ckpt_path(str(inference_ckpt), str(ckpt_root))


def test_resolve_ckpt_path_allows_inference_companion_when_opted_in(tmp_path):
    ckpt_root = tmp_path / "ckpts"
    run_dir = ckpt_root / "run123"
    run_dir.mkdir(parents=True)
    inference_ckpt = run_dir / "inference-anemoi-by_epoch-epoch_021-step_100000.ckpt"
    inference_ckpt.write_text("inference", encoding="utf-8")

    resolved = predict._resolve_ckpt_path(
        str(inference_ckpt),
        str(ckpt_root),
        allow_inference_companion=True,
    )

    assert resolved == str(inference_ckpt)


def test_predict_from_bundle_forwards_classic_sampling_args(monkeypatch):
    point_lres = 3
    point_hres = 4
    monkeypatch.setattr(predict, "open_bundle_dataset", lambda path: xr.Dataset())
    monkeypatch.setattr(predict, "find_missing_explicit_hres_inputs", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        predict,
        "load_inputs_from_bundle_numpy",
        lambda *args, **kwargs: (
            np.zeros((point_lres, 2), dtype=np.float32),
            np.zeros((point_hres, 1), dtype=np.float32),
            np.arange(point_lres, dtype=np.float32),
            np.arange(point_lres, dtype=np.float32) + 10,
            np.arange(point_hres, dtype=np.float32),
            np.arange(point_hres, dtype=np.float32) + 20,
        ),
    )
    monkeypatch.setattr(
        predict,
        "extract_target_from_bundle",
        lambda bundle_nc, weather_states: (None, 0),
    )

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"b": 0, "c": 1}}),
                        type("_In", (), {"name_to_index": {"z": 0}}),
                    ]
                },
            )
            self.model = type("_Model", (), {"output": type("_Out", (), {"name_to_index": {"b": 0, "c": 1}})})

    class _DummyDM:
        def __init__(self):
            self.data_indices = _Indices()
            self.ds_valid = type(
                "_DS",
                (),
                {
                    "data": type(
                        "_Data",
                        (),
                        {
                            "longitudes": [np.arange(point_lres), None, np.arange(point_hres)],
                            "latitudes": [np.arange(point_lres) + 10, None, np.arange(point_hres) + 20],
                        },
                    )
                },
            )

    model = _DummyModel(grid_hres=point_hres, n_states=2)
    dm = _DummyDM()
    classic = {
        "num_steps": 40,
        "sigma_max": 1000.0,
        "sigma_min": 0.03,
        "rho": 7.0,
        "sampler": "heun",
    }

    predict._predict_from_bundle(
        inference_model=model,
        datamodule=dm,
        device="cpu",
        bundle_nc="/tmp/fake.nc",
        member_index=0,
        extra_args=classic,
        precision="fp32",
        model_comm_group=None,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["extra_args"] == classic


def test_predict_from_bundle_rejects_nonzero_member_index(monkeypatch):
    monkeypatch.setattr(
        predict,
        "load_inputs_from_bundle_numpy",
        lambda *args, **kwargs: (
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"a": 0}}),
                        type("_In", (), {"name_to_index": {"z": 0}}),
                    ]
                },
            )
            self.model = type("_Model", (), {"output": type("_Out", (), {"name_to_index": {"a": 0}})})

    dm = type("_DummyDM", (), {"data_indices": _Indices()})()
    model = _DummyModel(grid_hres=3, n_states=1)

    try:
        predict._predict_from_bundle(
            inference_model=model,
            datamodule=dm,
            device="cpu",
            bundle_nc="/tmp/fake.nc",
            member_index=1,
            extra_args={},
            precision="fp32",
            model_comm_group=None,
        )
    except ValueError as exc:
        assert "single-member bundles" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nonzero bundle member index")


def test_fail_if_missing_truth_raises():
    with pytest.raises(SystemExit, match="Missing target truth `y`"):
        predict._fail_if_missing_truth(y=None, context="from-bundle")


def test_coerce_missing_truth_to_nan_requires_explicit_override():
    y_pred = np.ones((1, 1, 3, 2), dtype=np.float32)
    with pytest.raises(SystemExit, match="Missing target truth `y`"):
        predict._coerce_missing_truth_to_nan(
            y=None,
            y_pred=y_pred,
            context="from-bundle",
            allow_missing_target_unsafe=False,
        )


def test_coerce_missing_truth_to_nan_with_explicit_override():
    y_pred = np.ones((1, 1, 3, 2), dtype=np.float32)
    y, used_override = predict._coerce_missing_truth_to_nan(
        y=None,
        y_pred=y_pred,
        context="from-bundle",
        allow_missing_target_unsafe=True,
    )
    assert used_override is True
    assert y.shape == y_pred.shape
    assert np.isnan(y).all()


def test_predict_main_build_bundle_forwards_allow_missing_target_unsafe(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    fake_bundle_module = types.ModuleType("manual_inference.input_data_construction.bundle")

    def _fake_build_input_bundle_from_grib(**kwargs):
        captured.update(kwargs)
        return tmp_path / "bundle.nc"

    fake_bundle_module.build_input_bundle_from_grib = _fake_build_input_bundle_from_grib
    monkeypatch.setitem(
        sys.modules,
        "manual_inference.input_data_construction.bundle",
        fake_bundle_module,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "build-bundle",
            "--lres-sfc-grib",
            "lres_sfc.grib",
            "--lres-pl-grib",
            "lres_pl.grib",
            "--hres-grib",
            "hres.grib",
            "--allow-missing-target-unsafe",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    predict.main()

    assert captured["require_target_fields"] is False
    assert f"Saved bundle: {tmp_path / 'bundle.nc'}" in capsys.readouterr().out


def test_validate_output_path_rejects_nonempty_parent(tmp_path):
    out_dir = tmp_path / "run_a"
    out_dir.mkdir(parents=True)
    (out_dir / "already_here.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit, match="already exists and is not empty"):
        predict._validate_output_path(
            out_path=out_dir / "predictions.nc",
            allow_existing_output_dir=False,
        )


def test_validate_output_path_rejects_nested_run_layout(tmp_path):
    old_run = tmp_path / "old_run_20260309"
    (old_run / "logs").mkdir(parents=True)
    nested_out = old_run / "new_run_20260310" / "predictions.nc"

    with pytest.raises(SystemExit, match="Unsafe nested output path detected"):
        predict._validate_output_path(
            out_path=nested_out,
            allow_existing_output_dir=False,
        )


def test_resolve_output_weather_states_surface_plus_core_pl():
    weather_states = ["10u", "q_850", "msl", "u_850", "z_500", "t_850", "tcw"]

    selected, indices = resolve_output_weather_states(
        weather_states=weather_states,
        mode="surface-plus-core-pl",
    )

    assert selected == ["10u", "msl", "z_500", "t_850", "tcw"]
    assert indices == [0, 2, 4, 5, 6]


def test_build_predictions_dataset_slim_output_skips_member_views():
    x = np.zeros((1, 2, 3, 2), dtype=np.float32)
    y = np.zeros((1, 2, 4, 2), dtype=np.float32)
    y_pred = np.zeros((1, 2, 4, 2), dtype=np.float32)

    ds = build_predictions_dataset(
        x=x,
        y=y,
        y_pred=y_pred,
        lon_lres=np.arange(3, dtype=np.float32),
        lat_lres=np.arange(3, dtype=np.float32),
        lon_hres=np.arange(4, dtype=np.float32),
        lat_hres=np.arange(4, dtype=np.float32),
        weather_states=["10u", "t_850"],
        dates=None,
        member_ids=[1, 2],
        include_member_views=False,
    )

    assert "x_0" not in ds
    assert "y_0" not in ds
    assert "y_pred_0" not in ds
    assert tuple(ds["y_pred"].shape) == (1, 2, 4, 2)


def test_predict_main_from_bundle_writes_x_interp(tmp_path, monkeypatch):
    out_path = tmp_path / "predictions.nc"
    model = _DummyModel(grid_hres=4, n_states=2)

    monkeypatch.setattr(predict, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(predict, "_resolve_ckpt_path", lambda *args, **kwargs: str(tmp_path / "model.ckpt"))
    monkeypatch.setattr(predict, "_load_objects", lambda **kwargs: (model, object(), str(tmp_path), "demo-exp"))
    monkeypatch.setattr(
        predict,
        "_predict_from_bundle",
        lambda **kwargs: (
            np.array([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]], dtype=np.float32),
            np.zeros((1, 1, 4, 2), dtype=np.float32),
            np.ones((1, 1, 4, 2), dtype=np.float32),
            np.arange(3, dtype=np.float32),
            np.arange(3, dtype=np.float32) + 10,
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32) + 20,
            ["10u", "2t"],
            None,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "from-bundle",
            "--name-ckpt",
            "demo.ckpt",
            "--ckpt-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--bundle-nc",
            str(tmp_path / "bundle.nc"),
            "--out",
            str(out_path),
        ],
    )

    predict.main()

    with xr.open_dataset(out_path) as ds:
        assert "x_interp" in ds
        assert tuple(ds["x_interp"].shape) == (1, 1, 4, 2)
        np.testing.assert_allclose(
            ds["x_interp"].values,
            np.array([[[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]]], dtype=np.float32),
        )
        assert ds.attrs["x_interp_exported"] == 1


def test_predict_from_bundle_applies_output_subset(monkeypatch):
    point_lres = 3
    point_hres = 4
    requested_target_states = []
    monkeypatch.setattr(predict, "open_bundle_dataset", lambda path: xr.Dataset())
    monkeypatch.setattr(predict, "find_missing_explicit_hres_inputs", lambda *args, **kwargs: [])

    monkeypatch.setattr(
        predict,
        "load_inputs_from_bundle_numpy",
        lambda *args, **kwargs: (
            np.stack(
                [
                    np.full(point_lres, 1.0, dtype=np.float32),
                    np.full(point_lres, 2.0, dtype=np.float32),
                    np.full(point_lres, 3.0, dtype=np.float32),
                    np.full(point_lres, 4.0, dtype=np.float32),
                ],
                axis=1,
            ),
            np.zeros((point_hres, 1), dtype=np.float32),
            np.arange(point_lres, dtype=np.float32),
            np.arange(point_lres, dtype=np.float32) + 10,
            np.arange(point_hres, dtype=np.float32),
            np.arange(point_hres, dtype=np.float32) + 20,
        ),
    )

    def _fake_extract_target(bundle_nc, weather_states):
        requested_target_states.append(list(weather_states))
        target = np.zeros((point_hres, len(weather_states)), dtype=np.float32)
        return target, len(weather_states)

    monkeypatch.setattr(predict, "extract_target_from_bundle", _fake_extract_target)

    class _Indices:
        def __init__(self):
            self.data = type(
                "_Data",
                (),
                {
                    "input": [
                        type("_In", (), {"name_to_index": {"10u": 0, "u_850": 1, "t_850": 2, "msl": 3}}),
                        type("_In", (), {"name_to_index": {"z": 0}}),
                    ]
                },
            )
            self.model = type(
                "_Model",
                (),
                {"output": type("_Out", (), {"name_to_index": {"10u": 0, "u_850": 1, "t_850": 2, "msl": 3}})},
            )

    class _DummyDM:
        def __init__(self):
            self.data_indices = _Indices()
            self.ds_valid = type(
                "_DS",
                (),
                {
                    "data": type(
                        "_Data",
                        (),
                        {
                            "longitudes": [np.arange(point_lres), None, np.arange(point_hres)],
                            "latitudes": [np.arange(point_lres) + 10, None, np.arange(point_hres) + 20],
                        },
                    )
                },
            )

    model = _DummyModel(grid_hres=point_hres, n_states=4)
    dm = _DummyDM()

    out = predict._predict_from_bundle(
        inference_model=model,
        datamodule=dm,
        device="cpu",
        bundle_nc="/tmp/fake.nc",
        member_index=0,
        extra_args={},
        precision="fp32",
        model_comm_group=None,
        output_weather_state_mode="surface-plus-core-pl",
    )

    x_out, y_out, y_pred, *_coords, states, dates = out
    assert states == ["10u", "t_850", "msl"]
    assert requested_target_states == [["10u", "t_850", "msl"]]
    assert x_out.shape == (1, point_lres, 3)
    assert y_out.shape == (1, 1, point_hres, 3)
    assert y_pred.shape == (1, 1, point_hres, 3)
    assert dates is None


def test_predict_build_bundle_forwards_surface_extra_gribs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_build_input_bundle_from_grib(**kwargs):
        captured.update(kwargs)
        return tmp_path / "bundle.nc"

    fake_bundle_module = types.ModuleType("manual_inference.input_data_construction.bundle")
    fake_bundle_module.build_input_bundle_from_grib = _fake_build_input_bundle_from_grib
    monkeypatch.setitem(sys.modules, "manual_inference.input_data_construction.bundle", fake_bundle_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "build-bundle",
            "--lres-sfc-grib",
            "lres_input.grib",
            "--lres-sfc-extra-grib",
            "lres_input_tp.grib",
            "--lres-pl-grib",
            "lres_input.grib",
            "--hres-grib",
            "hres_sfc.grib",
            "--target-sfc-grib",
            "target_y.grib",
            "--target-sfc-extra-grib",
            "target_y_tp.grib",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    predict.main()

    assert captured["lres_sfc_extra_gribs"] == ["lres_input_tp.grib"]
    assert captured["target_sfc_extra_gribs"] == ["target_y_tp.grib"]


def test_predict_build_bundle_forwards_channel_subset_overrides(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_build_input_bundle_from_grib(**kwargs):
        captured.update(kwargs)
        return tmp_path / "bundle.nc"

    fake_bundle_module = types.ModuleType("manual_inference.input_data_construction.bundle")
    fake_bundle_module.build_input_bundle_from_grib = _fake_build_input_bundle_from_grib
    monkeypatch.setitem(sys.modules, "manual_inference.input_data_construction.bundle", fake_bundle_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "build-bundle",
            "--lres-sfc-grib",
            "lres_input.grib",
            "--lres-pl-grib",
            "lres_input.grib",
            "--hres-grib",
            "hres_sfc.grib",
            "--target-sfc-grib",
            "target_y.grib",
            "--lres-sfc-channels",
            "10u,10v,2t,msl",
            "--lres-pl-channels",
            "NONE",
            "--target-sfc-channels",
            "10u,10v,2t,msl",
            "--target-pl-channels",
            "NONE",
            "--out",
            str(tmp_path / "bundle.nc"),
        ],
    )

    predict.main()

    assert captured["lres_sfc_channels"] == ["10u", "10v", "2t", "msl"]
    assert captured["lres_pl_channels"] == []
    assert captured["target_sfc_channels"] == ["10u", "10v", "2t", "msl"]
    assert captured["target_pl_channels"] == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("10u,10v,2t,msl", ["10u", "10v", "2t", "msl"]),
        ("NONE", []),
        ("-", []),
        ("[]", []),
        ("empty", []),
    ],
)
def test_predict_parse_channel_subset_csv_supports_explicit_empty_override(raw, expected):
    assert predict._parse_channel_subset_csv(raw) == expected
