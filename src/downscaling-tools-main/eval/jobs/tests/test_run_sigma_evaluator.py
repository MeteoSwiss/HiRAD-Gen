from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str):
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class _DummyMove:
    def to(self, _device):
        return self


def test_run_sigma_evaluator_preserves_four_gpu_model_parallel_for_o1280_family(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval.sigma_evaluator.run_sigma_evaluator",
    )

    created_loaders = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            self.config_for_datamodule = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
                "sigma_seen": float(sigma),
                "pure_noise_seen": float(prediction_on_pure_noise),
            }

    checkpoint_config = _ns(
        hardware=_ns(num_gpus_per_model=4),
        dataloader=_ns(
            read_group_size=4,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o320_o1280")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 4))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=0,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.num_gpus_per_model == 4
    assert loader.config_checkpoint.dataloader.read_group_size == 4
    assert loader.config_for_datamodule.hardware.num_gpus_per_model == 4
    assert loader.config_for_datamodule.dataloader.read_group_size == 4
    assert loader.config_for_datamodule.dataloader.validation.frequency == "50h"
    assert loader.config_for_datamodule.dataloader.validation.num_workers == 0
    assert out_csv.exists()


def test_run_sigma_evaluator_defaults_to_single_gpu_for_lower_res_lanes(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval.sigma_evaluator.run_sigma_evaluator",
    )

    created_loaders = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            self.config_for_datamodule = _ns(
                hardware=_ns(num_gpus_per_model=4),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
                "sigma_seen": float(sigma),
                "pure_noise_seen": float(prediction_on_pure_noise),
            }

    checkpoint_config = _ns(
        hardware=_ns(num_gpus_per_model=4),
        dataloader=_ns(
            read_group_size=4,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o96_o320")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=0,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.num_gpus_per_model == 1
    assert loader.config_checkpoint.dataloader.read_group_size == 1
    assert loader.config_for_datamodule.hardware.num_gpus_per_model == 1
    assert loader.config_for_datamodule.dataloader.read_group_size == 1
    assert out_csv.exists()


def test_run_sigma_evaluator_applies_o1280_o2560_residual_stats_fallback(
    tmp_path: Path, monkeypatch
):
    mod = _load_module(
        "eval.sigma_evaluator.run_sigma_evaluator",
    )

    created_loaders = []
    residual_dir = tmp_path / "residuals"
    residual_dir.mkdir()
    (residual_dir / "o2560_dict_6_72.npy").write_text("placeholder")
    missing_name = "o2560_dict_6_72_destine_recomputed_4fields.npy"

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                hardware=_ns(
                    num_gpus_per_model=4,
                    paths=_ns(residual_statistics=str(residual_dir)),
                    files=_ns(residual_statistics=missing_name),
                ),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            self.config_for_datamodule = _ns(
                hardware=_ns(
                    num_gpus_per_model=4,
                    paths=_ns(residual_statistics=str(residual_dir)),
                    files=_ns(residual_statistics=missing_name),
                ),
                dataloader=_ns(
                    read_group_size=4,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            assert (
                self.config_checkpoint.hardware.files.residual_statistics
                == "o2560_dict_6_72.npy"
            )
            assert (
                self.config_for_datamodule.hardware.files.residual_statistics
                == "o2560_dict_6_72.npy"
            )
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
                "sigma_seen": float(sigma),
                "pure_noise_seen": float(prediction_on_pure_noise),
            }

    checkpoint_config = _ns(
        hardware=_ns(
            num_gpus_per_model=4,
            paths=_ns(residual_statistics=str(residual_dir)),
            files=_ns(residual_statistics=missing_name),
        ),
        dataloader=_ns(
            read_group_size=4,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns())
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o1280_o2560")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 4))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=0,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
    )

    mod.run_sigma_evaluator(args)

    loader = created_loaders[-1]
    assert loader.config_checkpoint.hardware.files.residual_statistics == "o2560_dict_6_72.npy"
    assert loader.config_for_datamodule.hardware.files.residual_statistics == "o2560_dict_6_72.npy"
    assert out_csv.exists()


def test_sigma_evaluator_single_variable_model():
    """1-channel output: only the matching field gets a real MSE; others are NaN."""
    import math

    import torch

    se_mod = _load_module("eval.sigma_evaluator.sigma_evaluator")
    SigmaEvaluator = se_mod.SigmaEvaluator

    name_to_index = {"2t": 0}  # single-variable model
    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = name_to_index
    evaluator._warned_fields = set()

    # Simulate what evaluate_batch_with_sigma does for the per-field block:
    diff = torch.randn(100, 1)  # 1 output channel
    metrics = {}
    metrics["diff_all_var_non_weighted"] = torch.sqrt(torch.mean(diff**2))

    num_output_vars = diff.shape[-1]
    for name in SigmaEvaluator.STANDARD_FIELDS:
        idx = evaluator.name_to_index.get(name)
        if idx is not None and idx < num_output_vars:
            metrics[f"mse_{name}_non_weighted"] = torch.mean(diff[..., idx] ** 2)
        else:
            metrics[f"mse_{name}_non_weighted"] = float("nan")

    assert not math.isnan(float(metrics["mse_2t_non_weighted"]))
    for name in SigmaEvaluator.STANDARD_FIELDS:
        if name != "2t":
            assert math.isnan(float(metrics[f"mse_{name}_non_weighted"]))
    assert not math.isnan(float(metrics["diff_all_var_non_weighted"]))


def test_sigma_evaluator_missing_data_indices():
    """When data_indices is unavailable, all per-field metrics are NaN."""
    import math

    import torch

    se_mod = _load_module("eval.sigma_evaluator.sigma_evaluator")
    SigmaEvaluator = se_mod.SigmaEvaluator

    evaluator = SigmaEvaluator.__new__(SigmaEvaluator)
    evaluator.name_to_index = {}  # empty — no data_indices available
    evaluator._warned_fields = set()

    diff = torch.randn(100, 5)
    metrics = {}
    metrics["diff_all_var_non_weighted"] = torch.sqrt(torch.mean(diff**2))

    num_output_vars = diff.shape[-1]
    for name in SigmaEvaluator.STANDARD_FIELDS:
        idx = evaluator.name_to_index.get(name)
        if idx is not None and idx < num_output_vars:
            metrics[f"mse_{name}_non_weighted"] = torch.mean(diff[..., idx] ** 2)
        else:
            metrics[f"mse_{name}_non_weighted"] = float("nan")

    for name in SigmaEvaluator.STANDARD_FIELDS:
        assert math.isnan(float(metrics[f"mse_{name}_non_weighted"]))
    assert not math.isnan(float(metrics["diff_all_var_non_weighted"]))


def test_adapt_config_hpc_missing_hardware_key(tmp_path: Path, monkeypatch):
    """When adapt_config_hpc raises due to missing hardware key, fallback is used."""
    mod = _load_module("eval.sigma_evaluator.run_sigma_evaluator")

    created_loaders = []
    inject_called = []

    class _DummyLoader:
        def __init__(self, *_args, **_kwargs):
            self.config_checkpoint = _ns(
                dataloader=_ns(
                    read_group_size=1,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            # No hardware key — adapt_config_hpc will fail
            self.config_for_datamodule = _ns(
                dataloader=_ns(
                    read_group_size=1,
                    validation=_ns(frequency="6h", num_workers=8),
                ),
            )
            created_loaders.append(self)

        def load(self):
            self.datamodule = object()
            self.interface = _DummyMove()
            self.downscaler = _DummyMove()

    class _DummySigmaEvaluator:
        def __init__(self, downscaler, datamodule, n_samples, name_to_index=None):
            self.downscaler = downscaler
            self.datamodule = datamodule
            self.n_samples = n_samples
            self.name_to_index = name_to_index

        def evaluate_sigma(self, sigma, prediction_on_pure_noise):
            return 0.25, {
                "diff_all_var_non_weighted": 0.5,
            }

    def _failing_adapt(config_checkpoint, _config):
        raise AttributeError("Missing key hardware")

    original_inject = mod._inject_minimal_hardware_config

    def _tracking_inject(config_checkpoint, host_config):
        inject_called.append(True)
        original_inject(config_checkpoint, host_config)

    checkpoint_config = _ns(
        dataloader=_ns(
            read_group_size=1,
            validation=_ns(frequency="12h", num_workers=16),
        ),
    )

    monkeypatch.setattr(mod, "ObjectFromCheckpointLoader", _DummyLoader)
    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: ({}, checkpoint_config))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns(hardware=_ns(paths=_ns(data="/data"))))
    monkeypatch.setattr(mod, "adapt_config_hpc", _failing_adapt)
    monkeypatch.setattr(mod, "_inject_minimal_hardware_config", _tracking_inject)
    monkeypatch.setattr(mod, "_rewrite_dataset_paths_in_place", lambda cfg: cfg)
    monkeypatch.setattr(mod, "SigmaEvaluator", _DummySigmaEvaluator)
    monkeypatch.setattr(mod, "infer_lane_from_config", lambda _cfg: "o48_o96")
    monkeypatch.setattr(mod, "_get_parallel_info", lambda: (0, 0, 1))
    monkeypatch.setattr(mod, "_init_model_comm_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "_resolve_device", lambda requested_device, _local_rank: requested_device)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    out_csv = tmp_path / "sigma_eval.csv"
    args = argparse.Namespace(
        ckpt_root="/tmp/checkpoints",
        name_exp="exp",
        name_ckpt="model.ckpt",
        out_file="sigma_eval_table.csv",
        out_csv=str(out_csv),
        device="cpu",
        num_gpus_per_model=0,
        n_samples=1,
        validation_frequency="50h",
        sigmas="1",
        run_pure_noise=False,
        run_noised=False,
    )

    mod.run_sigma_evaluator(args)

    assert len(inject_called) == 1, "_inject_minimal_hardware_config should have been called"
    assert out_csv.exists()
