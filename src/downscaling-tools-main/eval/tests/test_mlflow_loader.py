from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOADER_PY = ROOT / "mlflow" / "loader.py"
SPEC = importlib.util.spec_from_file_location("downscaling_tools_mlflow_loader", LOADER_PY)
assert SPEC is not None and SPEC.loader is not None
loader_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader_mod)


def _write_metric(path: Path, *rows: tuple[int, float, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{ts} {val} {step}\n" for ts, val, step in rows))


def _write_run(
    experiment_dir: Path,
    run_id: str,
    *,
    name: str,
    parent: str | None = None,
    metrics: dict[str, list[tuple[int, float, int]]],
) -> None:
    run_dir = experiment_dir / run_id
    (run_dir / "tags").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.yaml").write_text("artifact_uri: file:///tmp\n")
    (run_dir / "tags" / "mlflow.runName").write_text(f"{name}\n")
    if parent is not None:
        (run_dir / "tags" / "mlflow.parentRunId").write_text(f"{parent}\n")
    for rel_path, rows in metrics.items():
        _write_metric(run_dir / "metrics" / rel_path, *rows)


def test_load_run_family_bypasses_name_filters_and_merges_children(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    root_id = "8fd938156f6647d3894cd55dfc02698e"
    child_id = "11111111111111111111111111111111"

    _write_run(
        experiment_dir,
        root_id,
        name="to_delete",
        metrics={
            "train_weighted_mse_loss_epoch": [(1000, 2.0, 10)],
            "val_mse_metric/sfc_10u/1": [(1000, 5.0, 10)],
        },
    )
    _write_run(
        experiment_dir,
        child_id,
        name="00000000-0000-0000-0000-000000000000",
        parent=root_id,
        metrics={
            "train_weighted_mse_loss_epoch": [(2000, 1.0, 20)],
            "val_mse_metric/sfc_10u/1": [(2000, 3.0, 20)],
        },
    )

    assert loader_mod.load(experiment_dir) == {}

    runs = loader_mod.load_run_family(experiment_dir, root_id)

    assert list(runs) == ["to_delete"]
    run = runs["to_delete"]
    assert run["run_id"] == root_id
    assert run["max_step"] == 20
    assert run["metrics"]["train_weighted_mse_loss_epoch"]["steps"] == [10, 20]
    assert run["metrics"]["train_weighted_mse_loss_epoch"]["vals"] == [2.0, 1.0]
    assert run["metrics"]["val_mse_metric/sfc_10u/1"]["steps"] == [10, 20]


def test_load_run_family_uses_parent_when_child_run_requested(tmp_path: Path):
    experiment_dir = tmp_path / "experiment"
    root_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    child_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    _write_run(
        experiment_dir,
        root_id,
        name="family_root",
        metrics={"train_weighted_mse_loss_epoch": [(1000, 4.0, 10)]},
    )
    _write_run(
        experiment_dir,
        child_id,
        name="child_run",
        parent=root_id,
        metrics={"train_weighted_mse_loss_epoch": [(2000, 2.0, 20)]},
    )

    runs = loader_mod.load_run_family(experiment_dir, child_id)

    assert list(runs) == ["family_root"]
    run = runs["family_root"]
    assert run["run_id"] == root_id
    assert run["metrics"]["train_weighted_mse_loss_epoch"]["steps"] == [10, 20]


def test_load_run_family_recursive_chain(tmp_path: Path):
    """Verify a 4-deep chain (root→c1→c2→c3) is fully merged."""
    experiment_dir = tmp_path / "experiment"
    root_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"
    c1_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2"
    c2_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa3"
    c3_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa4"

    _write_run(experiment_dir, root_id, name="deep_chain",
               metrics={"train_loss": [(1000, 4.0, 10)]})
    _write_run(experiment_dir, c1_id, name="deep_chain_c1", parent=root_id,
               metrics={"train_loss": [(2000, 3.0, 20)]})
    _write_run(experiment_dir, c2_id, name="deep_chain_c2", parent=c1_id,
               metrics={"train_loss": [(3000, 2.0, 30)]})
    _write_run(experiment_dir, c3_id, name="deep_chain_c3", parent=c2_id,
               metrics={"train_loss": [(4000, 1.0, 40)]})

    # Request by deepest child — should walk up to root, then merge all 4
    runs = loader_mod.load_run_family(experiment_dir, c3_id)

    assert list(runs) == ["deep_chain"]
    run = runs["deep_chain"]
    assert run["run_id"] == root_id
    assert run["max_step"] == 40
    assert run["metrics"]["train_loss"]["steps"] == [10, 20, 30, 40]
    assert run["metrics"]["train_loss"]["vals"] == [4.0, 3.0, 2.0, 1.0]


def test_load_run_family_mid_chain_request(tmp_path: Path):
    """Request a mid-chain run; should still walk to root and get full history."""
    experiment_dir = tmp_path / "experiment"
    root_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1"
    c1_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2"
    c2_id = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb3"

    _write_run(experiment_dir, root_id, name="mid_test",
               metrics={"loss": [(1000, 5.0, 100)]})
    _write_run(experiment_dir, c1_id, name="mid_test_c1", parent=root_id,
               metrics={"loss": [(2000, 3.0, 200)]})
    _write_run(experiment_dir, c2_id, name="mid_test_c2", parent=c1_id,
               metrics={"loss": [(3000, 1.0, 300)]})

    # Request c1 (mid-chain) — should still walk to root, merge all 3
    runs = loader_mod.load_run_family(experiment_dir, c1_id)

    assert list(runs) == ["mid_test"]
    run = runs["mid_test"]
    assert run["run_id"] == root_id
    assert run["max_step"] == 300
    assert run["metrics"]["loss"]["steps"] == [100, 200, 300]
