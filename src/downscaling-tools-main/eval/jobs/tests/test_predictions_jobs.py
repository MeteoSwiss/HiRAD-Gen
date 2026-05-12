from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_predictions_parse_and_discover(tmp_path: Path):
    mod = _load_module(
        "gen25",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    p1 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    p2 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle_dup.nc"
    p3 = tmp_path / "eefo_o96_0001_date20230826_time0000_mem02_step024h_input_bundle.nc"
    p_bad = tmp_path / "not_a_bundle.nc"

    # Files must exist for rglob/stat based discovery.
    p1.write_text("a", encoding="utf-8")
    p2.write_text("b", encoding="utf-8")
    p3.write_text("c", encoding="utf-8")
    p_bad.write_text("x", encoding="utf-8")

    # Keep only parseable names in discovery; p2 does not match strict regex.
    k1 = mod.parse_bundle_key(p1)
    k3 = mod.parse_bundle_key(p3)
    assert k1 is not None
    assert k1.date == "20230826"
    assert k1.step == 24
    assert k1.member == 1
    assert k3 is not None
    assert mod.parse_bundle_key(p_bad) is None

    discovered = mod.discover_bundles(tmp_path)
    assert k1 in discovered
    assert k3 in discovered
    assert discovered[k1] == p1
    assert discovered[k3] == p3


def test_generate_predictions_discover_non_recursive_ignores_nested_cache(tmp_path: Path):
    mod = _load_module(
        "gen25_non_recursive",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    top = tmp_path / "top_level.txt"
    nested_dir = tmp_path / "bundles_with_y_proxy10"
    nested_dir.mkdir()
    nested_bundle = nested_dir / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    top.write_text("x", encoding="utf-8")
    nested_bundle.write_text("bundle", encoding="utf-8")

    discovered = mod.discover_bundles(tmp_path, recursive=False)
    assert discovered == {}


def test_generate_predictions_parse_int_list():
    mod = _load_module(
        "gen25_ints",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )
    assert mod.parse_int_list("3,1,2,2") == [1, 2, 3]
    assert mod.parse_int_list(" 10 , 1 , 5 ") == [1, 5, 10]


def test_generate_predictions_wait_for_rank0_write_done(tmp_path: Path):
    mod = _load_module(
        "gen25_wait_done",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    out_path = tmp_path / "predictions_20230826_step024.nc"

    def _writer():
        time.sleep(0.05)
        mod._rank0_done_marker(out_path).write_text("ok\n", encoding="utf-8")

    thread = threading.Thread(target=_writer)
    thread.start()
    try:
        mod._wait_for_rank0_write(
            out_path=out_path,
            global_rank=1,
            timeout_seconds=1,
            poll_seconds=0.01,
        )
    finally:
        thread.join()


def test_generate_predictions_wait_for_rank0_write_failure(tmp_path: Path):
    mod = _load_module(
        "gen25_wait_failed",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    out_path = tmp_path / "predictions_20230826_step024.nc"
    mod._rank0_failed_marker(out_path).write_text(
        "RuntimeError: boom\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Rank-0 write failed"):
        mod._wait_for_rank0_write(
            out_path=out_path,
            global_rank=1,
            timeout_seconds=1,
            poll_seconds=0.01,
        )


def test_generate_predictions_rejects_allow_missing_target(tmp_path: Path, monkeypatch):
    mod = _load_module(
        "gen25_reject_allow_missing_target",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )
    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy",
            "--allow-missing-target",
        ],
    )
    with pytest.raises(SystemExit, match="no longer supported"):
        mod.main()


def test_generate_predictions_allows_missing_target_unsafe(tmp_path: Path, monkeypatch):
    mod = _load_module(
        "gen25_allow_missing_target_unsafe",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_load_objects",
        lambda **kwargs: (object(), object(), "/tmp/dir_exp", "exp_name"),
    )
    monkeypatch.setattr(
        mod,
        "_compute_x_interp_for_export",
        lambda **kwargs: np.zeros((1, 1, 2, 3), dtype=np.float32),
    )

    def _fake_predict_from_bundle(**kwargs):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = None
        y_pred = np.ones((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    captured = {}

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 2}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["a", "b"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    def _fake_build_predictions_dataset(**kwargs):
        captured["y"] = kwargs["y"]
        return _FakeDS()

    monkeypatch.setattr(mod, "build_predictions_dataset", _fake_build_predictions_dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy",
            "--device",
            "cpu",
            "--allow-missing-target-unsafe",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    mod.main()

    assert "y" in captured
    assert captured["y"].shape == (1, 1, 2, 2)
    assert np.isnan(captured["y"]).all()


def test_generate_predictions_rejects_nonempty_out_dir(tmp_path: Path, monkeypatch):
    mod = _load_module(
        "gen25_reject_nonempty_out_dir",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )
    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "old.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy",
        ],
    )
    with pytest.raises(SystemExit, match="already exists and is not empty"):
        mod.main()


def test_generate_predictions_accepts_explicit_name_ckpt(monkeypatch, tmp_path: Path):
    mod = _load_module(
        "gen25_name_ckpt",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    load_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)

    def _fake_load_objects(**kwargs):
        load_calls.append(kwargs)
        return object(), object(), "/tmp/dir_exp", "exp_name"

    monkeypatch.setattr(mod, "_load_objects", _fake_load_objects)

    def _fake_predict_from_bundle(**kwargs):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 2}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["a", "b"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    monkeypatch.setattr(mod, "build_predictions_dataset", lambda **kwargs: _FakeDS())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--name-ckpt",
            "run123/anemoi-by_epoch-epoch_021-step_100000.ckpt",
            "--ckpt-root",
            str(tmp_path / "ckpt_root"),
            "--device",
            "cpu",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    mod.main()

    assert len(load_calls) == 1
    assert load_calls[0]["ckpt_path"].endswith(
        "run123/anemoi-by_epoch-epoch_021-step_100000.ckpt"
    )


def test_generate_predictions_bundle_pairs_override_dates_steps(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_bundle_pairs",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230828_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")
    (
        input_root
        / "eefo_o96_0001_date20230829_time0000_mem01_step048h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    predict_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_load_objects",
        lambda **kwargs: (object(), object(), "/tmp/dir_exp", "exp_name"),
    )

    def _fake_predict_from_bundle(**kwargs):
        predict_calls.append(Path(kwargs["bundle_nc"]).name)
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 2}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["a", "b"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    monkeypatch.setattr(mod, "build_predictions_dataset", lambda **kwargs: _FakeDS())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--device",
            "cpu",
            "--members",
            "1",
            "--dates",
            "20230826,20230827",
            "--steps",
            "24,72",
            "--bundle-pairs",
            "20230828:24,20230829:48",
        ],
    )

    mod.main()

    assert sorted(predict_calls) == [
        "eefo_o96_0001_date20230828_time0000_mem01_step024h_input_bundle.nc",
        "eefo_o96_0001_date20230829_time0000_mem01_step048h_input_bundle.nc",
    ]
    assert sorted(p.name for p in out_dir.glob("predictions_*.nc")) == [
        "predictions_20230828_step024.nc",
        "predictions_20230829_step048.nc",
    ]


def test_generate_predictions_rejects_existing_prediction_file(monkeypatch, tmp_path: Path):
    mod = _load_module(
        "gen25_reject_existing_prediction_file",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")
    (out_dir / "predictions_20230826_step024.nc").write_text("old", encoding="utf-8")

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_load_objects",
        lambda **kwargs: (object(), object(), "/tmp/dir_exp", "exp_name"),
    )

    def _fake_predict_from_bundle(**kwargs):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 2}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["a", "b"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    monkeypatch.setattr(mod, "build_predictions_dataset", lambda **kwargs: _FakeDS())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--allow-existing-out-dir",
            "--name-ckpt",
            "run123/anemoi-by_epoch-epoch_021-step_100000.ckpt",
            "--ckpt-root",
            str(tmp_path / "ckpt_root"),
            "--device",
            "cpu",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    with pytest.raises(SystemExit, match="Refusing to overwrite existing prediction file"):
        mod.main()


def test_generate_predictions_main_binds_cuda_device_and_gpu_override(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_main_distributed",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    set_device_calls = []
    init_group_calls = []
    load_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        mod.torch.cuda, "set_device", lambda idx: set_device_calls.append(idx)
    )
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 1, 4))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: f"cuda:{local}")
    monkeypatch.setattr(
        mod,
        "_init_model_comm_group",
        lambda device, rank, world: init_group_calls.append((device, rank, world))
        or "fake_group",
    )

    def _fake_load_objects(**kwargs):
        load_calls.append(kwargs)
        return object(), object(), "/tmp/dir_exp", "exp_name"

    monkeypatch.setattr(mod, "_load_objects", _fake_load_objects)
    monkeypatch.setattr(
        mod,
        "_compute_x_interp_for_export",
        lambda **kwargs: np.zeros((1, 1, 2, 2), dtype=np.float32),
    )


    def _fake_predict_from_bundle(**kwargs):
        x = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y = np.zeros((1, 1, 2, 2), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 2), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["a", "b"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 2}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["a", "b"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    monkeypatch.setattr(mod, "build_predictions_dataset", lambda **kwargs: _FakeDS())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--ckpt-root",
            str(tmp_path / "ckpt_root"),
            "--device",
            "cuda",
            "--num-gpus-per-model",
            "4",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    mod.main()

    assert set_device_calls == [1]
    assert init_group_calls == [("cuda:1", 0, 4)]
    assert len(load_calls) == 1
    assert load_calls[0]["device"] == "cuda:1"
    assert load_calls[0]["num_gpus_per_model_override"] == 4
    assert (out_dir / "predictions_20230826_step024.nc").exists()


def test_generate_predictions_rejects_world_size_mismatch(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_main_world_mismatch",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod.torch.cuda, "set_device", lambda idx: None)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cuda:0")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--device",
            "cuda",
            "--num-gpus-per-model",
            "4",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    with pytest.raises(SystemExit, match="Expected world_size=4"):
        mod.main()


def test_generate_predictions_passes_output_selection_and_slim_output(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_output_subset",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    predict_calls = []
    build_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_load_objects",
        lambda **kwargs: (object(), object(), "/tmp/dir_exp", "exp_name"),
    )
    monkeypatch.setattr(
        mod,
        "_compute_x_interp_for_export",
        lambda **kwargs: np.zeros((1, 1, 2, 3), dtype=np.float32),
    )


    def _fake_predict_from_bundle(**kwargs):
        predict_calls.append(kwargs)
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        y = np.zeros((1, 1, 2, 3), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 3), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["10u", "t_850", "msl"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 3}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type("_Arr", (), {"values": np.array(["10u", "t_850", "msl"], dtype=object)})()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    def _fake_build_predictions_dataset(**kwargs):
        build_calls.append(kwargs)
        return _FakeDS()

    monkeypatch.setattr(mod, "build_predictions_dataset", _fake_build_predictions_dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--device",
            "cpu",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
            "--output-weather-state-mode",
            "surface-plus-core-pl",
            "--slim-output",
        ],
    )

    mod.main()

    assert len(predict_calls) == 1
    assert predict_calls[0]["output_weather_state_mode"] == "surface-plus-core-pl"
    assert predict_calls[0]["output_weather_states"] is None
    assert len(build_calls) == 1
    assert build_calls[0]["x_interp"] is not None
    assert tuple(build_calls[0]["x_interp"].shape) == (1, 1, 2, 3)
    assert build_calls[0]["include_member_views"] is False
    assert (out_dir / "predictions_20230826_step024.nc").exists()


def test_generate_predictions_defaults_to_surface_plus_core_pl_and_slim(
    monkeypatch, tmp_path: Path
):
    mod = _load_module(
        "gen25_default_subset",
        ROOT / "eval/jobs/generate_predictions_25_files.py",
    )

    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    input_root.mkdir(parents=True, exist_ok=True)
    (
        input_root
        / "eefo_o96_0001_date20230826_time0000_mem01_step024h_input_bundle.nc"
    ).write_text("bundle", encoding="utf-8")

    predict_calls = []
    build_calls = []

    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_resolve_device", lambda requested, local: "cpu")
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_load_objects",
        lambda **kwargs: (object(), object(), "/tmp/dir_exp", "exp_name"),
    )
    monkeypatch.setattr(
        mod,
        "_compute_x_interp_for_export",
        lambda **kwargs: np.zeros((1, 1, 2, 3), dtype=np.float32),
    )

    def _fake_predict_from_bundle(**kwargs):
        predict_calls.append(kwargs)
        x = np.zeros((1, 1, 2, 3), dtype=np.float32)
        y = np.zeros((1, 1, 2, 3), dtype=np.float32)
        y_pred = np.zeros((1, 1, 2, 3), dtype=np.float32)
        lon_lres = np.zeros((2,), dtype=np.float32)
        lat_lres = np.zeros((2,), dtype=np.float32)
        lon_hres = np.zeros((2,), dtype=np.float32)
        lat_hres = np.zeros((2,), dtype=np.float32)
        weather_states = ["10u", "t_850", "msl"]
        return x, y, y_pred, lon_lres, lat_lres, lon_hres, lat_hres, weather_states, None

    monkeypatch.setattr(mod, "_predict_from_bundle", _fake_predict_from_bundle)

    class _FakeDS:
        def __init__(self):
            self.attrs = {}
            self.sizes = {"weather_state": 3}

        def assign_coords(self, **kwargs):
            return self

        def __getitem__(self, key):
            if key != "weather_state":
                raise KeyError(key)
            return type(
                "_Arr",
                (),
                {"values": np.array(["10u", "t_850", "msl"], dtype=object)},
            )()

        def __setitem__(self, key, value):
            return None

        def to_netcdf(self, path):
            Path(path).write_text("ok", encoding="utf-8")

        def close(self):
            return None

    def _fake_build_predictions_dataset(**kwargs):
        build_calls.append(kwargs)
        return _FakeDS()

    monkeypatch.setattr(mod, "build_predictions_dataset", _fake_build_predictions_dataset)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions_25_files.py",
            "--input-root",
            str(input_root),
            "--out-dir",
            str(out_dir),
            "--ckpt-id",
            "dummy_ckpt",
            "--device",
            "cpu",
            "--members",
            "1",
            "--steps",
            "24",
            "--dates",
            "20230826",
        ],
    )

    mod.main()

    assert len(predict_calls) == 1
    assert predict_calls[0]["output_weather_state_mode"] == "surface-plus-core-pl"
    assert len(build_calls) == 1
    assert build_calls[0]["x_interp"] is not None
    assert tuple(build_calls[0]["x_interp"].shape) == (1, 1, 2, 3)
    assert build_calls[0]["include_member_views"] is False
    assert (out_dir / "predictions_20230826_step024.nc").exists()


def test_autopilot_submit_and_state_parsing():
    mod = _load_module(
        "slurm_jobs_parse",
        ROOT / "eval/jobs/slurm_jobs.py",
    )

    assert mod.parse_submitted_job_id("Submitted batch job 123456") == "123456"
    assert mod.parse_sacct_state("UNKNOWN|\nRUNNING|\n", job_id="123456") == "RUNNING"
    assert mod.parse_squeue_state("") == (None, None)
    assert mod.parse_squeue_state(" running | Resources \n") == ("RUNNING", "Resources")


def test_slurm_jobs_marks_dependency_never_satisfied_failed():
    mod = _load_module(
        "slurm_jobs_state",
        ROOT / "eval/jobs/slurm_jobs.py",
    )

    def _run(cmd: list[str]) -> str:
        if cmd[0] == "squeue":
            return "PENDING|DependencyNeverSatisfied\n"
        raise AssertionError(f"Unexpected command: {cmd}")

    assert mod.job_state(_run, "123456") == "FAILED"


def test_autopilot_write_state(tmp_path: Path):
    mod = _load_module(
        "autopred_state",
        ROOT / "eval/jobs/autopilot_predictions.py",
    )
    state_file = tmp_path / "state.json"
    jobs = {
        "predict25": mod.JobTrack(
            name="predict25",
            script=Path("/tmp/predict.sbatch"),
            job_id="111",
            state="RUNNING",
            retries=0,
            max_retries=1,
        ),
        "eval25": mod.JobTrack(
            name="eval25",
            script=Path("/tmp/eval.sbatch"),
            dependency="predict25",
            job_id="222",
            state="PENDING",
            retries=0,
            max_retries=1,
        ),
    }
    mod._write_state(state_file, "manualabcd", mod.PHASE_PROXY, jobs)
    text = state_file.read_text(encoding="utf-8")
    assert "\"run_id\": \"manualabcd\"" in text
    assert f"\"phase\": \"{mod.PHASE_PROXY}\"" in text
    assert "\"predict25\"" in text
    assert "\"eval25\"" in text


def test_autopilot_rejects_unsafe_run_id(monkeypatch):
    mod = _load_module(
        "autopred_reject_unsafe_id",
        ROOT / "eval/jobs/autopilot_predictions.py",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_predictions.py",
            "--run-id",
            "bad/run",
        ],
    )
    with pytest.raises(SystemExit, match="unsafe characters"):
        mod.main()


def test_autopilot_rejects_eval_root_that_looks_like_run(monkeypatch, tmp_path: Path):
    mod = _load_module(
        "autopred_reject_eval_root",
        ROOT / "eval/jobs/autopilot_predictions.py",
    )
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "jobs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autopilot_predictions.py",
            "--run-id",
            "manualabcd",
            "--eval-root",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit, match="looks like a run directory"):
        mod.main()


def test_launch_predictions_eval_suite_dry_run_generates_scripts(tmp_path: Path):
    import subprocess
    import uuid

    run_id = f"manual_{uuid.uuid4().hex[:8]}"
    script = ROOT / "eval/jobs/launch_predictions_eval_suite.sh"
    eval_root = tmp_path / "eval_root"
    run_dir = eval_root / run_id
    generated_dir = run_dir / "jobs"

    out = subprocess.check_output(
        [
            str(script),
            "--run-id",
            run_id,
            "--eval-root",
            str(eval_root),
            "--ckpt-id",
            "4a5b2f1b24b84c52872bfcec1410b00f",
            "--dry-run",
        ],
        text=True,
    )
    assert "Dry run. Scripts written" in out

    pred = generated_dir / f"predict25_{run_id}.sbatch"
    evl = generated_dir / f"eval25_{run_id}.sbatch"
    assert pred.exists()
    assert evl.exists()

    pred_text = pred.read_text(encoding="utf-8")
    evl_text = evl.read_text(encoding="utf-8")
    assert "generate_predictions_25_files.py" in pred_text
    assert "export DATA_DIR=/home/mlx/ai-ml/datasets/" in pred_text
    assert "export OUTPUT=/ec/res4/scratch/ecm5702/aifs" in pred_text
    assert f"--out-dir {run_dir}/predictions" in pred_text
    assert "python -m eval.run" in evl_text
    assert f"--eval-root {run_dir}/eval" in evl_text
    assert "Expected 25 prediction files" in evl_text


def test_launch_proxy_eval_dry_run_uses_repo_owned_spectra_helper(tmp_path: Path):
    import subprocess
    import uuid

    run_id = f"proxy_{uuid.uuid4().hex[:8]}"
    script = ROOT / "eval/jobs/launch_proxy_eval.sh"
    eval_root = tmp_path / "eval_root"
    run_dir = eval_root / run_id
    generated_dir = run_dir / "jobs"

    out = subprocess.check_output(
        [
            str(script),
            "--run-id",
            run_id,
            "--eval-root",
            str(eval_root),
            "--ckpt-id",
            "4a5b2f1b24b84c52872bfcec1410b00f",
            "--write-scoreboard-artifacts",
            "--dry-run",
        ],
        text=True,
    )
    assert "Dry run. Scripts written" in out

    evl = generated_dir / f"eval_proxy_{run_id}.sbatch"
    assert evl.exists()

    evl_text = evl.read_text(encoding="utf-8")
    helper_path = ROOT / "eval/jobs/templates/predictions_dir_spectra.py"
    assert str(helper_path) in evl_text
    assert "/dev/docs/scratch/predictions_dir_spectra.py" not in evl_text
    assert "preserving scheduler success" not in evl_text
    assert "Proxy TC comparison failed with exit code" in evl_text
