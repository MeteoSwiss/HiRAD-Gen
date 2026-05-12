from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

from manual_inference import checkpoints as mod


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_loader_allows_missing_truncation_data(monkeypatch, tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    run_dir = ckpt_dir / "exp"
    run_dir.mkdir(parents=True)
    (run_dir / "model.ckpt").write_text("ckpt")
    (run_dir / "inference-model.ckpt").write_text("inference")

    checkpoint_cfg = _ns(hardware=_ns(paths={}))
    checkpoint = {"hyper_parameters": {"config": checkpoint_cfg, "metadata": {}}}
    inference_model = _ns(graph_data={"graph": 1})

    monkeypatch.setattr(mod, "get_checkpoint", lambda *_args, **_kwargs: (checkpoint, checkpoint_cfg))
    monkeypatch.setattr(mod, "instantiate_config", lambda: _ns(hardware=_ns(paths={})))
    monkeypatch.setattr(mod, "adapt_config_hpc", lambda config_checkpoint, _config: config_checkpoint)
    monkeypatch.setattr(mod, "to_omegaconf", lambda cfg: cfg)
    monkeypatch.setattr(mod.torch, "cuda", _ns(is_available=lambda: False))
    monkeypatch.setattr(mod.torch, "load", lambda *_args, **_kwargs: inference_model)
    monkeypatch.setattr(mod, "get_datamodule", lambda *_args, **_kwargs: _ns(statistics={}, data_indices={}, supporting_arrays={}))
    monkeypatch.setattr(mod, "get_interface", lambda *_args, **_kwargs: "interface")
    monkeypatch.setattr(mod, "get_downscaler", lambda *_args, **_kwargs: "downscaler")

    loader = mod.ObjectFromCheckpointLoader(str(ckpt_dir), "exp", "model.ckpt")
    loader.load()

    assert loader.graph_data == {"graph": 1}
    assert loader.truncation_data is None
    assert loader.interface == "interface"
    assert loader.downscaler == "downscaler"


def test_get_interface_falls_back_to_downscaling_model_interface(monkeypatch):
    fake_interface_mod = ModuleType("anemoi.models.interface")

    class _FallbackInterface:
        def __init__(
            self,
            *,
            config,
            graph_data,
            statistics,
            data_indices,
            interp_data,
            metadata,
        ):
            self.kwargs = {
                "config": config,
                "graph_data": graph_data,
                "statistics": statistics,
                "data_indices": data_indices,
                "interp_data": interp_data,
                "metadata": metadata,
            }

    fake_interface_mod.DownscalingModelInterface = _FallbackInterface
    monkeypatch.setitem(sys.modules, "anemoi.models.interface", fake_interface_mod)

    datamodule = _ns(statistics={"s": 1}, data_indices={"i": 2}, supporting_arrays={"up": 1, "down": 2})
    checkpoint = {"hyper_parameters": {"metadata": {"m": 3}}}

    interface = mod.get_interface(
        config_checkpoint={"cfg": 1},
        datamodule=datamodule,
        graph_data={"g": 1},
        truncation_data=None,
        checkpoint=checkpoint,
    )

    assert isinstance(interface, _FallbackInterface)
    assert interface.kwargs["config"] == {"cfg": 1}
    assert interface.kwargs["graph_data"] == {"g": 1}
    assert interface.kwargs["interp_data"] == {"up": 1, "down": 2}


def test_get_downscaler_falls_back_to_graph_downscaler(monkeypatch, tmp_path):
    fallback_mod = ModuleType("anemoi.training.train.downscaler")

    class _FallbackDownscaler:
        called = None

        @classmethod
        def load_from_checkpoint(cls, path, strict=False, **kwargs):
            cls.called = {"path": path, "strict": strict, "kwargs": kwargs}
            return "downscaler"

    fallback_mod.GraphDownscaler = _FallbackDownscaler
    broken_tasks_mod = ModuleType("anemoi.training.train.tasks.downscaler")
    monkeypatch.setitem(sys.modules, "anemoi.training.train.tasks.downscaler", broken_tasks_mod)
    monkeypatch.setitem(sys.modules, "anemoi.training.train.downscaler", fallback_mod)

    datamodule = _ns(data_indices={"i": 1}, statistics={"s": 2})
    out = mod.get_downscaler(
        dir_exp=str(tmp_path),
        name_exp="exp",
        name_ckpt="model.ckpt",
        checkpoint={"hyper_parameters": {"metadata": {"m": 3}}},
        config_checkpoint={"cfg": 4},
        graph_data={"g": 5},
        datamodule=datamodule,
        truncation_data=None,
    )

    assert out == "downscaler"
    assert _FallbackDownscaler.called["strict"] is False
    assert "truncation_data" not in _FallbackDownscaler.called["kwargs"]
    assert "supporting_arrays" not in _FallbackDownscaler.called["kwargs"]
