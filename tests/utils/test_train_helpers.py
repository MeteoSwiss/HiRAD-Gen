import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from hirad.utils.train_helpers import (
    calculate_patch_per_iter,
    check_model_health,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    init_mlflow,
    is_time_for_periodic_task,
    set_patch_shape,
    set_seed,
    update_learning_rate,
)


############################################################################
#                            set_patch_shape                               #
############################################################################


class TestSetPatchShape:
    """Tests for set_patch_shape."""

    def test_patch_equals_image_disables_patching(self):
        use_patching, img, patch = set_patch_shape((128, 128), (128, 128))
        assert use_patching is False
        assert img == (128, 128)
        assert patch == (128, 128)

    def test_none_patch_defaults_to_image(self):
        use_patching, img, patch = set_patch_shape((128, 256), (None, None))
        assert use_patching is False
        assert patch == (128, 256)

    def test_patch_larger_than_image_clamped(self):
        use_patching, img, patch = set_patch_shape((64, 64), (128, 128))
        assert use_patching is False
        assert patch == (64, 64)

    def test_valid_square_patch_enables_patching(self):
        use_patching, img, patch = set_patch_shape((256, 256), (64, 64))
        assert use_patching is True
        assert patch == (64, 64)

    def test_patch_not_multiple_of_32_raises(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            set_patch_shape((256, 256), (50, 50))

    def test_rectangular_patch_raises(self):
        with pytest.raises(NotImplementedError, match="Rectangular patch"):
            set_patch_shape((256, 256), (64, 128))

    def test_img_shape_returned_unchanged(self):
        _, img, _ = set_patch_shape((100, 200), (None, None))
        assert img == (100, 200)

    def test_patch_32_is_valid(self):
        use_patching, _, patch = set_patch_shape((256, 256), (32, 32))
        assert use_patching is True
        assert patch == (32, 32)


############################################################################
#                              set_seed                                    #
############################################################################


class TestSetSeed:
    """Tests for set_seed."""

    def test_reproducibility(self):
        set_seed(42)
        a_np = np.random.rand(5)
        a_torch = torch.rand(5)

        set_seed(42)
        b_np = np.random.rand(5)
        b_torch = torch.rand(5)

        np.testing.assert_array_equal(a_np, b_np)
        torch.testing.assert_close(a_torch, b_torch)

    def test_different_ranks_give_different_seeds(self):
        set_seed(0)
        a_np = np.random.rand(100)
        a_torch = torch.rand(100)

        set_seed(1)
        b_np = np.random.rand(100)
        b_torch = torch.rand(100)

        assert not np.array_equal(a_np, b_np)
        assert not torch.allclose(a_torch, b_torch)

    def test_large_rank_wraps(self):
        """Ranks larger than 2^31 should still work due to modulo."""
        large_rank = (1 << 31) + 5
        set_seed(large_rank)
        a_np = np.random.rand(5)
        a_torch = torch.rand(5)

        set_seed(5)
        b_np = np.random.rand(5)
        b_torch = torch.rand(5)
        # rank % (1<<31) should give 5 in both cases
        np.testing.assert_array_equal(a_np, b_np)
        torch.testing.assert_close(a_torch, b_torch)


############################################################################
#                   compute_num_accumulation_rounds                        #
############################################################################


class TestComputeNumAccumulationRounds:
    """Tests for compute_num_accumulation_rounds."""

    def test_single_gpu_no_accumulation(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=16, batch_size_per_gpu=16, world_size=1
        )
        assert batch_gpu_total == 16
        assert num_rounds == 1

    def test_multi_gpu_no_accumulation(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=32, batch_size_per_gpu=8, world_size=4
        )
        assert batch_gpu_total == 8
        assert num_rounds == 1

    def test_accumulation_rounds(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=64, batch_size_per_gpu=8, world_size=2
        )
        assert batch_gpu_total == 32
        assert num_rounds == 4

    def test_none_batch_size_per_gpu_defaults_to_total(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=32, batch_size_per_gpu=None, world_size=2
        )
        assert batch_gpu_total == 16
        assert num_rounds == 1

    def test_batch_size_per_gpu_larger_than_total_clamped(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=16, batch_size_per_gpu=64, world_size=2
        )
        assert batch_gpu_total == 8
        assert num_rounds == 1

    def test_invalid_batch_sizes_raise(self):
        """total_batch_size not divisible properly should raise ValueError."""
        with pytest.raises(ValueError, match="total_batch_size must be equal"):
            compute_num_accumulation_rounds(
                total_batch_size=17, batch_size_per_gpu=4, world_size=2
            )

    def test_world_size_1_full_accumulation(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=64, batch_size_per_gpu=16, world_size=1
        )
        assert batch_gpu_total == 64
        assert num_rounds == 4

    def test_exact_division(self):
        batch_gpu_total, num_rounds = compute_num_accumulation_rounds(
            total_batch_size=128, batch_size_per_gpu=16, world_size=4
        )
        assert batch_gpu_total == 32
        assert num_rounds == 2
        assert 16 * 2 * 4 == 128


############################################################################
#                      handle_and_clip_gradients                           #
############################################################################


class TestHandleAndClipGradients:
    """Tests for handle_and_clip_gradients."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple linear model with computed gradients."""
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        return model

    def test_nan_gradients_replaced(self):
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        # Inject NaN into gradient
        model.weight.grad[0, 0] = float("nan")
        handle_and_clip_gradients(model)
        assert torch.isfinite(model.weight.grad).all()

    def test_inf_gradients_replaced(self):
        model = torch.nn.Linear(4, 2, bias=False)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        model.weight.grad[0, 0] = float("inf")
        model.weight.grad[1, 0] = float("-inf")
        handle_and_clip_gradients(model)
        assert torch.isfinite(model.weight.grad).all()

    def test_gradient_clipping(self, simple_model):
        # Set a large gradient
        simple_model.weight.grad.fill_(100.0)
        handle_and_clip_gradients(simple_model, grad_clip_threshold=1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            simple_model.parameters(), float("inf")
        )
        assert grad_norm <= 1.0 + 1e-6

    def test_no_clipping_when_none(self, simple_model):
        original_grad = simple_model.weight.grad.clone()
        handle_and_clip_gradients(simple_model, grad_clip_threshold=None)
        torch.testing.assert_close(simple_model.weight.grad, original_grad)

    def test_params_without_grad_skipped(self):
        """Parameters without gradients should not cause errors."""
        model = torch.nn.Linear(4, 2, bias=True)
        # Only weight has grad, bias does not
        model.weight.grad = torch.randn_like(model.weight)
        model.bias.grad = None
        handle_and_clip_gradients(model)  # Should not raise


############################################################################
#                         check_model_health                               #
############################################################################


class TestCheckModelHealth:
    """Tests for check_model_health."""

    @pytest.fixture
    def logger(self):
        return MagicMock()

    def test_healthy_model_returns_true(self, logger):
        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        assert check_model_health(model, step=0, logger=logger) is True
        logger.warning.assert_not_called()

    def test_nan_weights_returns_false(self, logger):
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        result = check_model_health(model, step=5, logger=logger)
        assert result is False
        logger.warning.assert_called_once()
        assert "Weights" in logger.warning.call_args[0][0]

    def test_inf_weights_returns_false(self, logger):
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model.weight[0, 0] = float("inf")
        result = check_model_health(model, step=3, logger=logger)
        assert result is False
        logger.warning.assert_called_once()
        assert "Weights" in logger.warning.call_args[0][0]

    def test_nan_gradients_returns_false(self, logger):
        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        model.weight.grad[0, 0] = float("nan")
        result = check_model_health(model, step=10, logger=logger)
        assert result is False
        assert "Gradients" in logger.warning.call_args[0][0]

    def test_inf_gradients_returns_false(self, logger):
        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        model.weight.grad[1, 1] = float("inf")
        result = check_model_health(model, step=7, logger=logger)
        assert result is False
        assert "Gradients" in logger.warning.call_args[0][0]

    def test_no_grad_params_healthy(self, logger):
        """Parameters without gradients should not cause false negatives."""
        model = torch.nn.Linear(4, 2, bias=True)
        # No backward called, so no gradients
        result = check_model_health(model, step=0, logger=logger)
        assert result is True

    def test_step_number_in_warning(self, logger):
        model = torch.nn.Linear(4, 2)
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        check_model_health(model, step=42, logger=logger)
        assert "42" in logger.warning.call_args[0][0]


############################################################################
#                      is_time_for_periodic_task                           #
############################################################################


class TestIsTimeForPeriodicTask:
    """Tests for is_time_for_periodic_task."""

    def test_exact_frequency_match(self):
        assert is_time_for_periodic_task(
            cur_nimg=100, freq=100, done=False, batch_size=10, rank=0
        ) is True

    def test_within_batch_of_frequency(self):
        # cur_nimg=105, freq=100 => 105 % 100 = 5 < batch_size=10
        assert is_time_for_periodic_task(
            cur_nimg=105, freq=100, done=False, batch_size=10, rank=0
        ) is True

    def test_not_time_yet(self):
        # cur_nimg=50, freq=100 => 50 % 100 = 50 >= batch_size=10
        assert is_time_for_periodic_task(
            cur_nimg=50, freq=100, done=False, batch_size=10, rank=0
        ) is False

    def test_done_always_returns_true(self):
        assert is_time_for_periodic_task(
            cur_nimg=50, freq=100, done=True, batch_size=10, rank=0
        ) is True

    def test_rank_0_only_blocks_other_ranks(self):
        assert is_time_for_periodic_task(
            cur_nimg=100, freq=100, done=False, batch_size=10,
            rank=1, rank_0_only=True,
        ) is False

    def test_rank_0_only_allows_rank_0(self):
        assert is_time_for_periodic_task(
            cur_nimg=100, freq=100, done=False, batch_size=10,
            rank=0, rank_0_only=True,
        ) is True

    def test_rank_0_only_false_allows_any_rank(self):
        assert is_time_for_periodic_task(
            cur_nimg=100, freq=100, done=False, batch_size=10,
            rank=3, rank_0_only=False,
        ) is True

    def test_done_overrides_rank_0_only(self):
        """done=True should return True even for non-zero ranks with rank_0_only."""
        assert is_time_for_periodic_task(
            cur_nimg=50, freq=100, done=True, batch_size=10,
            rank=2, rank_0_only=True,
        ) is False  # rank_0_only check happens first

    def test_zero_cur_nimg(self):
        # 0 % freq = 0 < batch_size => True
        assert is_time_for_periodic_task(
            cur_nimg=0, freq=100, done=False, batch_size=10, rank=0
        ) is True

    def test_batch_size_equals_freq(self):
        # Every step should trigger when batch_size >= freq
        assert is_time_for_periodic_task(
            cur_nimg=37, freq=100, done=False, batch_size=100, rank=0
        ) is True


############################################################################
#                            init_mlflow                                   #
############################################################################


class TestInitMlflow:
    """Tests for init_mlflow."""

    @pytest.fixture
    def base_cfg(self):
        """Minimal config DictConfig for init_mlflow."""
        return OmegaConf.create(
            {
                "logging": {
                    "uri": "http://mlflow-server:5000",
                    "experiment_name": "test_experiment",
                    "run_name": "test_run",
                },
            }
        )

    @pytest.fixture
    def cfg_no_uri(self):
        """Config with logging.uri set to None."""
        return OmegaConf.create(
            {
                "logging": {
                    "uri": None,
                    "experiment_name": "test_experiment",
                    "run_name": "test_run",
                },
            }
        )

    @pytest.fixture
    def dist_rank0_single(self):
        """DistributedManager mock: rank 0, world_size 1."""
        dist = MagicMock()
        dist.rank = 0
        dist.world_size = 1
        dist._local_rank = 0
        return dist

    @pytest.fixture
    def dist_rank0_multi(self):
        """DistributedManager mock: rank 0, world_size 4."""
        dist = MagicMock()
        dist.rank = 0
        dist.world_size = 4
        dist._local_rank = 0
        return dist

    @pytest.fixture
    def dist_rank0_large(self):
        """DistributedManager mock: rank 0, world_size 8 (>4)."""
        dist = MagicMock()
        dist.rank = 0
        dist.world_size = 8
        dist._local_rank = 0
        return dist

    @pytest.fixture
    def mock_mlflow(self):
        """Patch mlflow and related utilities used in init_mlflow."""
        with patch("hirad.utils.train_helpers.mlflow") as m_mlflow, \
             patch("hirad.utils.train_helpers.get_env_info") as m_env, \
             patch("hirad.utils.train_helpers.flatten_dict") as m_flat:
            # get_env_info returns (dict, git_diff_string)
            m_env.return_value = ({"pkg": {"version": "1.0"}}, "diff contents")
            # flatten_dict passthrough
            m_flat.side_effect = lambda x: x
            # active_run mock
            mock_run = MagicMock()
            mock_run.info.run_id = "new-run-id-123"
            m_mlflow.active_run.return_value = mock_run
            yield {
                "mlflow": m_mlflow,
                "get_env_info": m_env,
                "flatten_dict": m_flat,
            }

    # --- Rank 0, fresh run (no existing run_id.txt) ---

    def test_rank0_fresh_run_sets_tracking_uri(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_tracking_uri.assert_called_once_with(
            "http://mlflow-server:5000"
        )

    def test_rank0_fresh_run_sets_experiment(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_experiment.assert_called_once_with(
            experiment_name="test_experiment"
        )

    def test_rank0_fresh_run_starts_with_run_name(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].start_run.assert_called_once_with(
            run_name="test_run", log_system_metrics=True
        )

    def test_rank0_fresh_run_saves_run_id(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        run_id_file = tmp_path / "run_id.txt"
        assert run_id_file.exists()
        assert run_id_file.read_text() == "new-run-id-123"

    def test_rank0_fresh_run_logs_params(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_params.assert_called_once()

    def test_rank0_fresh_run_logs_env_info(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_dict.assert_any_call(
            {"pkg": {"version": "1.0"}}, "environment.json"
        )

    def test_rank0_fresh_run_logs_git_diff(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_text.assert_called_once_with(
            "diff contents", "git_diff.txt"
        )

    def test_rank0_fresh_run_logs_config(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_dict.assert_any_call(base_cfg, "config.json")

    # --- Rank 0, no git diff ---

    def test_rank0_no_git_diff_skips_log_text(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        mock_mlflow["get_env_info"].return_value = ({"pkg": {}}, "")
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_text.assert_not_called()

    # --- Rank 0, resuming from checkpoint (run_id.txt exists) ---

    def test_rank0_resume_reads_run_id(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        (tmp_path / "run_id.txt").write_text("existing-run-id-456")
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].start_run.assert_called_once_with(
            run_id="existing-run-id-456", log_system_metrics=True
        )

    def test_rank0_resume_does_not_log_params(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        (tmp_path / "run_id.txt").write_text("existing-run-id-456")
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].log_params.assert_not_called()

    def test_rank0_resume_does_not_overwrite_run_id(self, base_cfg, dist_rank0_single, mock_mlflow, tmp_path):
        (tmp_path / "run_id.txt").write_text("existing-run-id-456")
        init_mlflow(base_cfg, dist_rank0_single, write_dir=str(tmp_path))
        # File content should remain unchanged
        assert (tmp_path / "run_id.txt").read_text() == "existing-run-id-456"

    # --- URI handling ---

    def test_rank0_none_uri_skips_set_tracking(self, cfg_no_uri, dist_rank0_single, mock_mlflow, tmp_path):
        init_mlflow(cfg_no_uri, dist_rank0_single, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_tracking_uri.assert_not_called()

    # --- System metrics node ID ---

    def test_rank0_small_world_sets_node_id(self, base_cfg, dist_rank0_multi, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_multi, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].system_metrics.set_system_metrics_node_id.assert_called_once_with(
            "node-0"
        )

    def test_rank0_large_world_disables_system_metrics(self, base_cfg, dist_rank0_large, mock_mlflow, tmp_path):
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist_rank0_large, write_dir=str(tmp_path))
        # world_size > 4: log_system_metrics=False
        mock_mlflow["mlflow"].start_run.assert_called_once()
        _, kwargs = mock_mlflow["mlflow"].start_run.call_args
        assert kwargs["log_system_metrics"] is False

    def test_rank0_small_world_enables_system_metrics(self, base_cfg, dist_rank0_multi, mock_mlflow, tmp_path):
        init_mlflow(base_cfg, dist_rank0_multi, write_dir=str(tmp_path))
        _, kwargs = mock_mlflow["mlflow"].start_run.call_args
        assert kwargs["log_system_metrics"] is True

    # --- Distributed barrier for large world_size ---

    def test_large_world_calls_barrier(self, base_cfg, dist_rank0_large, mock_mlflow, tmp_path):
        with patch("hirad.utils.train_helpers.torch") as m_torch:
            init_mlflow(base_cfg, dist_rank0_large, write_dir=str(tmp_path))
            m_torch.distributed.barrier.assert_called_once()

    def test_small_world_skips_barrier(self, base_cfg, dist_rank0_multi, mock_mlflow, tmp_path):
        with patch("hirad.utils.train_helpers.torch") as m_torch:
            init_mlflow(base_cfg, dist_rank0_multi, write_dir=str(tmp_path))
            m_torch.distributed.barrier.assert_not_called()

    # --- Sub-node MLflow activation (non-rank-0 local rank 0) ---

    def test_sub_node_rank4_local0_starts_run(self, base_cfg, mock_mlflow, tmp_path):
        """rank=4, _local_rank=0, world_size=8 should activate sub mlflow."""
        dist = MagicMock()
        dist.rank = 4
        dist.world_size = 8
        dist._local_rank = 0
        (tmp_path / "run_id.txt").write_text("existing-run-id-456")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].start_run.assert_called_once_with(
            run_id="existing-run-id-456", log_system_metrics=True
        )

    def test_sub_node_sets_correct_node_id(self, base_cfg, mock_mlflow, tmp_path):
        """rank=4, world_size=8 should set node id to 'node-1'."""
        dist = MagicMock()
        dist.rank = 4
        dist.world_size = 8
        dist._local_rank = 0
        (tmp_path / "run_id.txt").write_text("run-id")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].system_metrics.set_system_metrics_node_id.assert_called_once_with(
            "node-1"
        )

    def test_rank1_large_world_activates_sub_mlflow(self, base_cfg, mock_mlflow, tmp_path):
        """rank=1, world_size=8 should activate sub mlflow (special case)."""
        dist = MagicMock()
        dist.rank = 1
        dist.world_size = 8
        dist._local_rank = 1
        (tmp_path / "run_id.txt").write_text("run-id")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        # rank=1 special case: node_id is "node-0"
        mock_mlflow["mlflow"].system_metrics.set_system_metrics_node_id.assert_called_once_with(
            "node-0"
        )
        mock_mlflow["mlflow"].start_run.assert_called_once_with(
            run_id="run-id", log_system_metrics=True
        )

    def test_rank1_small_world_skips_sub_mlflow(self, base_cfg, mock_mlflow, tmp_path):
        """rank=1, world_size=4 should NOT activate sub mlflow."""
        dist = MagicMock()
        dist.rank = 1
        dist.world_size = 4
        dist._local_rank = 1
        init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        # rank != 0, not local_rank 0, world_size <= 4 => no start_run
        mock_mlflow["mlflow"].start_run.assert_not_called()

    def test_non_local_rank0_non_rank1_skips_sub_mlflow(self, base_cfg, mock_mlflow, tmp_path):
        """rank=2, _local_rank=2, world_size=8 should not activate sub mlflow."""
        dist = MagicMock()
        dist.rank = 2
        dist.world_size = 8
        dist._local_rank = 2
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].start_run.assert_not_called()

    def test_non_local_rank0_non_node_0_skips_sub_mlflow(self, base_cfg, mock_mlflow, tmp_path):
        """rank=5, _local_rank=1, world_size=8 should not activate sub mlflow."""
        dist = MagicMock()
        dist.rank = 5
        dist.world_size = 8
        dist._local_rank = 1
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].start_run.assert_not_called()

    def test_sub_node_sets_tracking_uri(self, base_cfg, mock_mlflow, tmp_path):
        """Sub-node should set tracking URI when configured."""
        dist = MagicMock()
        dist.rank = 4
        dist.world_size = 8
        dist._local_rank = 0
        (tmp_path / "run_id.txt").write_text("run-id")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_tracking_uri.assert_called_with(
            "http://mlflow-server:5000"
        )

    def test_sub_node_none_uri_skips_starting_mlflow(self, cfg_no_uri, mock_mlflow, tmp_path):
        """Sub-node should skip starting mlflow when URI is None."""
        dist = MagicMock()
        dist.rank = 4
        dist.world_size = 8
        dist._local_rank = 0
        (tmp_path / "run_id.txt").write_text("run-id")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(cfg_no_uri, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_tracking_uri.assert_not_called()
        mock_mlflow["mlflow"].start_run.assert_not_called()

    def test_sub_node_sets_experiment(self, base_cfg, mock_mlflow, tmp_path):
        """Sub-node should set the experiment name."""
        dist = MagicMock()
        dist.rank = 4
        dist.world_size = 8
        dist._local_rank = 0
        (tmp_path / "run_id.txt").write_text("run-id")
        with patch("hirad.utils.train_helpers.torch"):
            init_mlflow(base_cfg, dist, write_dir=str(tmp_path))
        mock_mlflow["mlflow"].set_experiment.assert_called_with(
            experiment_name="test_experiment"
        )


############################################################################
#                        update_learning_rate                              #
############################################################################


class TestUpdateLearningRate:
    """Tests for update_learning_rate."""

    @staticmethod
    def _make_optimizer(lr, num_groups=1):
        """Create a simple SGD optimizer with `num_groups` param groups."""
        params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(num_groups)]
        optimizer = torch.optim.SGD(
            [{"params": [p], "lr": lr} for p in params]
        )
        return optimizer

    # ------------------------------------------------------------------
    # Rampup phase  (cur_nimg < lr_rampup)
    # ------------------------------------------------------------------
    def test_rampup_halfway(self):
        """At half the rampup period the LR should be lr * 0.5."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=1.0, lr_decay_rate=1, cur_nimg=500)
        assert result == pytest.approx(0.01 * 0.5)

    def test_rampup_quarter(self):
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.02, lr_rampup=2000,
                                      lr_decay=1.0, lr_decay_rate=1, cur_nimg=500)
        assert result == pytest.approx(0.02 * 0.25)

    def test_rampup_at_zero(self):
        """At cur_nimg=0 the LR should be 0 during rampup."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=1.0, lr_decay_rate=1, cur_nimg=0)
        assert result == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # Rampup boundary  (cur_nimg == lr_rampup)
    # ------------------------------------------------------------------
    def test_rampup_exact_boundary(self):
        """At the exact rampup boundary LR should equal base lr (no decay yet)."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=0.5, lr_decay_rate=500, cur_nimg=1000)
        # rampup factor = min(1000/1000, 1) = 1  →  lr = 0.01
        # decay exponent = (1000 - 1000) // 500 = 0  →  0.5^0 = 1
        assert result == pytest.approx(0.01)

    # ------------------------------------------------------------------
    # Post-rampup with decay
    # ------------------------------------------------------------------
    def test_decay_one_step(self):
        """One decay step after rampup."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=0.5, lr_decay_rate=500, cur_nimg=1500)
        # rampup clamped at 1, decay = 0.5 ^ ((1500-1000)//500) = 0.5^1
        assert result == pytest.approx(0.01 * 0.5)

    def test_decay_two_steps(self):
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=0.5, lr_decay_rate=500, cur_nimg=2000)
        # decay = 0.5 ^ ((2000-1000)//500) = 0.5^2 = 0.25
        assert result == pytest.approx(0.01 * 0.25)

    def test_decay_partial_step_floors(self):
        """Decay uses integer division, so partial steps are floored."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=0.5, lr_decay_rate=500, cur_nimg=1499)
        # (1499-1000)//500 = 0  →  no decay yet
        assert result == pytest.approx(0.01)

    # ------------------------------------------------------------------
    # No rampup  (lr_rampup == 0)
    # ------------------------------------------------------------------
    def test_no_rampup_applies_decay_to_existing_lr(self):
        """When lr_rampup=0 the base lr is NOT overwritten;
        decay is applied to the optimizer's current lr."""
        opt = self._make_optimizer(lr=0.04)
        result = update_learning_rate(opt, lr=999,  # ignored for the set step
                                      lr_rampup=0, lr_decay=0.5,
                                      lr_decay_rate=100, cur_nimg=100)
        # g["lr"] stays 0.04 (rampup branch skipped), then *= 0.5^(100//100) = 0.5
        assert result == pytest.approx(0.04 * 0.5)

    def test_no_rampup_no_decay(self):
        """lr_rampup=0, lr_decay=1.0 → LR unchanged."""
        opt = self._make_optimizer(lr=0.03)
        result = update_learning_rate(opt, lr=999, lr_rampup=0,
                                      lr_decay=1.0, lr_decay_rate=100, cur_nimg=500)
        assert result == pytest.approx(0.03)

    # ------------------------------------------------------------------
    # No decay  (lr_decay == 1.0)
    # ------------------------------------------------------------------
    def test_rampup_without_decay(self):
        """Rampup works independently of decay when decay=1.0."""
        opt = self._make_optimizer(lr=0.1)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=1.0, lr_decay_rate=500, cur_nimg=2000)
        assert result == pytest.approx(0.01)

    # ------------------------------------------------------------------
    # Multiple param groups
    # ------------------------------------------------------------------
    def test_multiple_param_groups(self):
        """All param groups are updated; return value is the last group's LR."""
        opt = self._make_optimizer(lr=0.1, num_groups=3)
        result = update_learning_rate(opt, lr=0.01, lr_rampup=1000,
                                      lr_decay=1.0, lr_decay_rate=1, cur_nimg=500)
        expected = 0.01 * 0.5
        for g in opt.param_groups:
            assert g["lr"] == pytest.approx(expected)
        assert result == pytest.approx(expected)

    # ------------------------------------------------------------------
    # Successive calls (simulating a training loop)
    # ------------------------------------------------------------------
    def test_successive_calls_during_rampup(self):
        """LR should grow linearly across successive rampup calls."""
        opt = self._make_optimizer(lr=0.1)
        lr, rampup = 0.01, 1000
        lrs = []
        for step in range(0, 1001, 200):
            lrs.append(
                update_learning_rate(opt, lr=lr, lr_rampup=rampup,
                                     lr_decay=1.0, lr_decay_rate=1, cur_nimg=step)
            )
        expected = [lr * min(s / rampup, 1) for s in range(0, 1001, 200)]
        for got, exp in zip(lrs, expected):
            assert got == pytest.approx(exp)

    def test_successive_calls_with_decay(self):
        """LR should decrease in staircase fashion after rampup."""
        opt = self._make_optimizer(lr=0.1)
        lr, rampup, decay, rate = 0.01, 0, 0.9, 100
        prev_lr = None
        for step in [0, 50, 100, 150, 200]:
            # reset optimizer lr before each call since no rampup means
            # the function mutates the existing lr multiplicatively
            for g in opt.param_groups:
                g["lr"] = lr
            cur = update_learning_rate(opt, lr=lr, lr_rampup=rampup,
                                       lr_decay=decay, lr_decay_rate=rate,
                                       cur_nimg=step)
            expected = lr * decay ** (step // rate)
            assert cur == pytest.approx(expected)


############################################################################
#                       calculate_patch_per_iter                           #
############################################################################


class TestCalculatePatchPerIter:
    """Tests for calculate_patch_per_iter."""

    # ------------------------------------------------------------------
    # max_patch_per_gpu is None / falsy  →  single iteration
    # ------------------------------------------------------------------
    def test_no_max_returns_single_element(self):
        """When max_patch_per_gpu is None, return [patch_num]."""
        assert calculate_patch_per_iter(4, None, 1) == [4]

    def test_no_max_zero_returns_single_element(self):
        """When max_patch_per_gpu is 0 (falsy), return [patch_num]."""
        assert calculate_patch_per_iter(8, 0, 2) == [8]

    def test_no_max_patch_num_one(self):
        assert calculate_patch_per_iter(1, None, 1) == [1]

    # ------------------------------------------------------------------
    # max_patch_per_gpu provided – fits in a single iteration
    # ------------------------------------------------------------------
    def test_single_iter_exact_fit(self):
        """patch_num fits exactly within max_patch_per_gpu."""
        # max_patch_num_per_iter = min(4, 8//2) = 4 → 1 iteration
        assert calculate_patch_per_iter(4, 8, 2) == [4]

    def test_single_iter_max_exceeds_patch_num(self):
        """max allows more patches than needed; still one iteration."""
        # max_patch_num_per_iter = min(2, 16//1) = 2 → 1 iteration
        assert calculate_patch_per_iter(2, 16, 1) == [2]

    # ------------------------------------------------------------------
    # max_patch_per_gpu provided – requires multiple iterations
    # ------------------------------------------------------------------
    def test_even_split(self):
        """patch_num divides evenly into iterations."""
        # max_patch_num_per_iter = min(8, 4//1) = 4 → 2 iterations of 4
        assert calculate_patch_per_iter(8, 4, 1) == [4, 4]

    def test_uneven_split(self):
        """Last iteration gets fewer patches."""
        # max_patch_num_per_iter = min(7, 4//1) = 4
        # iterations = ceil(7/4) = 2 → [4, 3]
        assert calculate_patch_per_iter(7, 4, 1) == [4, 3]

    def test_three_iterations(self):
        """Requires three iterations with a remainder."""
        # max_patch_num_per_iter = min(10, 4//1) = 4
        # iterations = ceil(10/4) = 3 → [4, 4, 2]
        assert calculate_patch_per_iter(10, 4, 1) == [4, 4, 2]

    def test_patch_num_one_less_than_max(self):
        # max_patch_num_per_iter = min(3, 4//1) = 3 → single iteration
        assert calculate_patch_per_iter(3, 4, 1) == [3]

    def test_patch_num_one_more_than_max(self):
        # max_patch_num_per_iter = min(5, 4//1) = 4
        # iterations = ceil(5/4) = 2 → [4, 1]
        assert calculate_patch_per_iter(5, 4, 1) == [4, 1]

    # ------------------------------------------------------------------
    # batch_size_per_gpu interaction
    # ------------------------------------------------------------------
    def test_batch_size_reduces_max_per_iter(self):
        """Larger batch size reduces the effective max patches per iter."""
        # max_patch_num_per_iter = min(6, 8//4) = 2
        # iterations = ceil(6/2) = 3 → [2, 2, 2]
        assert calculate_patch_per_iter(6, 8, 4) == [2, 2, 2]

    def test_batch_size_equals_max(self):
        """batch_size_per_gpu == max_patch_per_gpu → 1 patch per iter."""
        # max_patch_num_per_iter = min(3, 4//4) = 1
        # iterations = ceil(3/1) = 3 → [1, 1, 1]
        assert calculate_patch_per_iter(3, 4, 4) == [1, 1, 1]

    # ------------------------------------------------------------------
    # Validation / edge cases
    # ------------------------------------------------------------------
    def test_max_less_than_batch_raises(self):
        """max_patch_per_gpu < batch_size_per_gpu should raise."""
        with pytest.raises(ValueError, match="max_patch_per_gpu"):
            calculate_patch_per_iter(4, 2, 4)

    def test_sum_equals_patch_num(self):
        """Sum of returned list must always equal patch_num."""
        for patch_num in range(1, 20):
            for batch in [1, 2, 4]:
                for max_ppg in [batch, batch * 2, batch * 3, batch * 5]:
                    result = calculate_patch_per_iter(patch_num, max_ppg, batch)
                    assert sum(result) == patch_num, (
                        f"patch_num={patch_num}, max_ppg={max_ppg}, batch={batch}: "
                        f"sum({result}) = {sum(result)} != {patch_num}"
                    )

    def test_no_element_exceeds_max(self):
        """No single iteration should exceed max_patch_num_per_iter."""
        for patch_num in range(1, 20):
            for batch in [1, 2, 4]:
                for max_ppg in [batch, batch * 2, batch * 3]:
                    max_per_iter = min(patch_num, max_ppg // batch)
                    result = calculate_patch_per_iter(patch_num, max_ppg, batch)
                    assert all(r <= max_per_iter for r in result), (
                        f"patch_num={patch_num}, max_ppg={max_ppg}, batch={batch}: "
                        f"{result} has element > {max_per_iter}"
                    )

