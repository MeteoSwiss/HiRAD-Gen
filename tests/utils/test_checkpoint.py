import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hirad.utils.checkpoint import (
    _get_checkpoint_filename,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
#  Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_mock_manager(model_parallel_rank: int = 0, group_names=()):
    """Return a mock DistributedManager with the given parallel rank."""
    mgr = MagicMock()
    mgr.group_names = group_names
    mgr.group_rank.return_value = model_parallel_rank
    return mgr


@pytest.fixture(autouse=True)
def _patch_distributed():
    """Patch DistributedManager for every test so we never touch real dist."""
    mgr = _make_mock_manager(model_parallel_rank=0, group_names=())
    with patch(
        "hirad.utils.checkpoint.DistributedManager"
    ) as MockDM:
        MockDM.is_initialized.return_value = True
        MockDM.return_value = mgr
        yield MockDM


class _SimpleModel(torch.nn.Module):
    """Tiny model used in save / load round-trip tests."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


############################################################################
#                       _get_checkpoint_filename                           #
############################################################################


class TestGetCheckpointFilenameWithIndex:
    """When an explicit index is supplied."""

    def test_returns_correct_filename(self, tmp_path):
        result = _get_checkpoint_filename(str(tmp_path), index=3)
        expected = str(tmp_path.resolve() / "checkpoint.0.3.pt")
        assert result == expected

    def test_custom_base_name(self, tmp_path):
        result = _get_checkpoint_filename(str(tmp_path), base_name="model", index=0)
        expected = str(tmp_path.resolve() / "model.0.0.pt")
        assert result == expected

    def test_custom_model_type(self, tmp_path):
        result = _get_checkpoint_filename(
            str(tmp_path), index=1, model_type="hirad"
        )
        expected = str(tmp_path.resolve() / "checkpoint.0.1.hirad")
        assert result == expected

    def test_saving_flag_ignored_when_index_given(self, tmp_path):
        result_save = _get_checkpoint_filename(str(tmp_path), index=5, saving=True)
        result_load = _get_checkpoint_filename(str(tmp_path), index=5, saving=False)
        assert result_save == result_load


class TestGetCheckpointFilenameNoIndex:
    """When no index is supplied (auto-detect from existing files)."""

    def test_no_existing_files_returns_index_zero(self, tmp_path):
        result = _get_checkpoint_filename(str(tmp_path))
        expected = str(tmp_path.resolve() / "checkpoint.0.0.pt")
        assert result == expected

    def test_loads_latest_when_files_exist(self, tmp_path):
        # Create fake checkpoint files with indices 0, 1, 2
        for i in range(3):
            (tmp_path / f"checkpoint.0.{i}.pt").touch()
        result = _get_checkpoint_filename(str(tmp_path), saving=False)
        expected = str(tmp_path.resolve() / "checkpoint.0.2.pt")
        assert result == expected

    def test_saving_increments_latest_index(self, tmp_path):
        for i in range(3):
            (tmp_path / f"checkpoint.0.{i}.pt").touch()
        result = _get_checkpoint_filename(str(tmp_path), saving=True)
        expected = str(tmp_path.resolve() / "checkpoint.0.3.pt")
        assert result == expected

    def test_non_contiguous_indices_picks_largest(self, tmp_path):
        for i in [0, 5, 10]:
            (tmp_path / f"checkpoint.0.{i}.pt").touch()
        result = _get_checkpoint_filename(str(tmp_path), saving=False)
        expected = str(tmp_path.resolve() / "checkpoint.0.10.pt")
        assert result == expected


class TestGetCheckpointFilenameModelParallel:
    """Ensure model-parallel rank is embedded correctly."""

    def test_model_parallel_rank_in_filename(self, tmp_path, _patch_distributed):
        mgr = _make_mock_manager(model_parallel_rank=3, group_names=("model_parallel",))
        _patch_distributed.return_value = mgr
        result = _get_checkpoint_filename(str(tmp_path), index=0)
        expected = str(tmp_path.resolve() / "checkpoint.3.0.pt")
        assert result == expected


############################################################################
#                           save_checkpoint                                #
############################################################################


class TestSaveCheckpointDirectory:
    """Test directory creation behaviour of save_checkpoint."""

    def test_creates_directory_if_missing(self, tmp_path):
        out = str(tmp_path / "new_dir" / "sub")
        model = _SimpleModel()
        save_checkpoint(out, model=model, epoch=0)
        assert Path(out).is_dir()

    def test_existing_directory_is_fine(self, tmp_path):
        model = _SimpleModel()
        save_checkpoint(str(tmp_path), model=model, epoch=0)
        # Just ensure no exception is raised
        assert Path(tmp_path).is_dir()


class TestSaveCheckpointModel:
    """Test model state-dict saving."""

    def test_model_checkpoint_file_created(self, tmp_path):
        model = _SimpleModel()
        save_checkpoint(str(tmp_path), model=model, epoch=0)
        expected = tmp_path.resolve() / f"{model.__class__.__name__}.0.0.pt"
        assert expected.exists()

    def test_model_checkpoint_contains_state_dict(self, tmp_path):
        model = _SimpleModel()
        save_checkpoint(str(tmp_path), model=model, epoch=0)
        file_name = tmp_path.resolve() / f"{model.__class__.__name__}.0.0.pt"
        state = torch.load(file_name, map_location="cpu")
        assert "linear.weight" in state
        assert "linear.bias" in state


class TestSaveCheckpointTraining:
    """Test optimizer / scheduler / scaler / metadata saving."""

    def test_optimizer_state_saved(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(str(tmp_path), optimizer=opt, epoch=1)
        ckpt_file = tmp_path.resolve() / "checkpoint.0.1.pt"
        assert ckpt_file.exists()
        ckpt = torch.load(ckpt_file, map_location="cpu")
        assert "optimizer_state_dict" in ckpt

    def test_epoch_saved(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(str(tmp_path), optimizer=opt, epoch=5)
        ckpt_file = tmp_path.resolve() / "checkpoint.0.5.pt"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        assert ckpt["epoch"] == 5

    def test_metadata_saved(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        meta = {"loss": 0.42, "note": "test run"}
        save_checkpoint(str(tmp_path), optimizer=opt, epoch=0, metadata=meta)
        ckpt_file = tmp_path.resolve() / "checkpoint.0.0.pt"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        assert ckpt["metadata"] == meta

    def test_no_training_objects_no_checkpoint_file(self, tmp_path):
        """If only a model is provided, no training checkpoint should be created."""
        model = _SimpleModel()
        save_checkpoint(str(tmp_path), model=model, epoch=0)
        training_ckpt = tmp_path.resolve() / "checkpoint.0.0.pt"
        assert not training_ckpt.exists()

    def test_scheduler_state_saved(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        save_checkpoint(str(tmp_path), optimizer=opt, scheduler=sched, epoch=0)
        ckpt_file = tmp_path.resolve() / "checkpoint.0.0.pt"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        assert "scheduler_state_dict" in ckpt

    def test_scaler_state_saved(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.cuda.amp.GradScaler()
        save_checkpoint(str(tmp_path), optimizer=opt, scaler=scaler, epoch=0)
        ckpt_file = tmp_path.resolve() / "checkpoint.0.0.pt"
        ckpt = torch.load(ckpt_file, map_location="cpu")
        assert "scaler_state_dict" in ckpt


############################################################################
#                           load_checkpoint                                #
############################################################################


class TestLoadCheckpointMissingDir:
    """Loading from a non-existent directory should return 0 gracefully."""

    def test_returns_zero_for_missing_dir(self, tmp_path):
        result = load_checkpoint(str(tmp_path / "nonexistent"))
        assert result == 0


class TestLoadCheckpointModel:
    """Round-trip save/load of model state dicts."""

    def test_model_weights_restored(self, tmp_path):
        model = _SimpleModel()
        # Freeze initial weights for comparison
        original_weight = model.linear.weight.data.clone()
        save_checkpoint(str(tmp_path), model=model, epoch=0)

        # Mutate weights so we can confirm they get restored
        with torch.no_grad():
            model.linear.weight.fill_(0.0)
        assert not torch.equal(model.linear.weight.data, original_weight)

        load_checkpoint(str(tmp_path), model=model, epoch=0)
        assert torch.equal(model.linear.weight.data, original_weight)

    def test_missing_model_file_is_graceful(self, tmp_path):
        """If model checkpoint doesn't exist, load should not crash."""
        tmp_path.mkdir(exist_ok=True)
        model = _SimpleModel()
        # No save happened, but directory exists – should warn and skip
        result = load_checkpoint(str(tmp_path), model=model, epoch=0)
        # Should still return 0 (no training checkpoint either)
        assert result == 0


class TestLoadCheckpointTraining:
    """Round-trip save/load of training state (optimizer, scheduler, epoch, metadata)."""

    def test_optimizer_restored(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        # Take a step so state is non-trivial
        loss = model.linear(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()
        original_state = {k: v for k, v in opt.state_dict().items()}

        save_checkpoint(str(tmp_path), optimizer=opt, epoch=1)

        # Create a fresh optimizer
        opt2 = torch.optim.SGD(model.parameters(), lr=0.01)
        load_checkpoint(str(tmp_path), optimizer=opt2, epoch=1)
        assert opt2.state_dict()["param_groups"] == original_state["param_groups"]

    def test_epoch_returned(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        save_checkpoint(str(tmp_path), optimizer=opt, epoch=7)
        loaded_epoch = load_checkpoint(str(tmp_path), epoch=7)
        assert loaded_epoch == 7

    def test_metadata_restored(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        meta = {"lr": 1e-3, "run_id": "abc"}
        save_checkpoint(str(tmp_path), optimizer=opt, epoch=0, metadata=meta)

        restored_meta = {}
        load_checkpoint(str(tmp_path), epoch=0, metadata_dict=restored_meta)
        assert restored_meta == meta

    def test_missing_checkpoint_file_returns_zero(self, tmp_path):
        """If training checkpoint file doesn't exist, return 0."""
        tmp_path.mkdir(exist_ok=True)
        result = load_checkpoint(str(tmp_path), epoch=99)
        assert result == 0


class TestSaveLoadRoundTrip:
    """Full round-trip tests combining model + training state."""

    def test_full_round_trip(self, tmp_path):
        model = _SimpleModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5)
        meta = {"step": 100}

        # Forward + backward to populate optimizer state
        loss = model.linear(torch.randn(2, 4)).sum()
        loss.backward()
        opt.step()
        sched.step()

        original_weight = model.linear.weight.data.clone()

        save_checkpoint(
            str(tmp_path), model=model, optimizer=opt,
            scheduler=sched, epoch=3, metadata=meta,
        )

        # Reset everything
        with torch.no_grad():
            model.linear.weight.fill_(0.0)
        opt2 = torch.optim.SGD(model.parameters(), lr=0.01)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=5)
        restored_meta = {}

        loaded_epoch = load_checkpoint(
            str(tmp_path), model=model, optimizer=opt2,
            scheduler=sched2, epoch=3, metadata_dict=restored_meta,
        )

        assert loaded_epoch == 3
        assert torch.equal(model.linear.weight.data, original_weight)
        assert restored_meta == meta
        assert opt2.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
        assert sched2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
