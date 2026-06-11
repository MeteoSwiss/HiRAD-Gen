from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[2]
PLOT_PY = ROOT / "mlflow" / "plot.py"
SPEC = importlib.util.spec_from_file_location("downscaling_tools_mlflow_plot", PLOT_PY)
assert SPEC is not None and SPEC.loader is not None
plot_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_mod)


def _demo_runs() -> dict[str, dict[str, object]]:
    return {
        "demo_run": {
            "run_id": "demo_run_id",
            "max_step": 2,
            "metrics": {
                "train_weighted_mse_loss_epoch": {"steps": [1, 2], "vals": [2.0, 1.0]},
                "val_weighted_mse_loss_epoch": {"steps": [1, 2], "vals": [2.5, 1.2]},
                "lr-AdamW": {"steps": [1, 2], "vals": [1e-3, 5e-4]},
                "val_mse_metric/all/1": {"steps": [1, 2], "vals": [5.0, 2.0]},
                "val_mse_metric/sfc_10u/1": {"steps": [1, 2], "vals": [1.5, 0.8]},
                "val_mse_metric/pl_t/1": {"steps": [1, 2], "vals": [3.0, 1.4]},
                "val_mse_metric/z_500/1": {"steps": [1, 2], "vals": [2.2, 1.1]},
            },
        }
    }


def test_all_val_metric_keys_excludes_aggregate_metric():
    keys = plot_mod._all_val_metric_keys(_demo_runs())

    assert keys == [
        "val_mse_metric/pl_t/1",
        "val_mse_metric/sfc_10u/1",
        "val_mse_metric/z_500/1",
    ]


def test_plot_all_writes_all_standard_loss_artifacts(tmp_path: Path):
    plot_mod.plot_all(_demo_runs(), output_dir=tmp_path)

    assert (tmp_path / "key_vars.png").exists()
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "all_vars.png").exists()
