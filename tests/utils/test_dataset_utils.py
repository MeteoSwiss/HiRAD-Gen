import numpy as np
import pytest
import torch

from hirad.utils.dataset_utils import GridData, regrid_icon_to_rotlatlon


############################################################################
#                        regrid_icon_to_rotlatlon                          #
############################################################################


class TestRegridIconToRotlatlon:
    """Tests for the regrid_icon_to_rotlatlon function."""

    @pytest.fixture
    def simple_regrid_inputs(self):
        """Create simple inputs for regrid testing."""
        n_unstructured = 20
        n_target = 6  # 3 x 2 grid
        n_stencil = 3

        data = torch.arange(n_unstructured, dtype=torch.float32)
        indices = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8],
             [9, 10, 11], [12, 13, 14], [15, 16, 17]],
            dtype=torch.long,
        )
        weights = torch.tensor(
            [[0.5, 0.3, 0.2]] * n_target, dtype=torch.float32
        )
        return data, indices, weights

    def test_output_shape_2d(self, simple_regrid_inputs):
        data, indices, weights = simple_regrid_inputs
        nx, ny = 3, 2
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=nx, ny=ny)
        assert result.shape == (ny, nx)

    def test_output_shape_with_batch(self, simple_regrid_inputs):
        _, indices, weights = simple_regrid_inputs
        batch, channels, n_unstructured = 2, 4, 20
        nx, ny = 3, 2
        data = torch.randn(batch, channels, n_unstructured)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=nx, ny=ny)
        assert result.shape == (batch, channels, ny, nx)

    def test_output_shape_with_channels(self, simple_regrid_inputs):
        _, indices, weights = simple_regrid_inputs
        channels, n_unstructured = 5, 20
        nx, ny = 3, 2
        data = torch.randn(channels, n_unstructured)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=nx, ny=ny)
        assert result.shape == (channels, ny, nx)

    def test_uniform_weights_give_mean(self):
        """When all weights are equal the result should be the mean of stencil values."""
        n_target = 4
        n_stencil = 3
        data = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]], dtype=torch.long)
        weights = torch.full((n_target, n_stencil), 1.0 / n_stencil)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=2, ny=2)
        expected_vals = torch.tensor([2.0, 2.0, 20.0, 20.0]).reshape(2, 2)
        torch.testing.assert_close(result, expected_vals)

    def test_single_weight_selects_value(self):
        """Weight concentrated on one index should select that value."""
        data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
        weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=1, ny=1)
        assert result.item() == pytest.approx(10.0)

    def test_output_values_with_batch_and_channels(self):
        """Test that values are correctly computed with batch and channel dimensions."""
        batch, channels, n_unstructured = 2, 3, 20
        nx, ny = 3, 2
        data = torch.arange(batch * channels * n_unstructured, dtype=torch.float32).reshape(batch, channels, n_unstructured)
        indices = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8],
             [9, 10, 11], [12, 13, 14], [15, 16, 17]],
            dtype=torch.long,
        )
        weights = torch.tensor(
            [[0.5, 0.3, 0.2]] * (nx * ny), dtype=torch.float32
        )
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=nx, ny=ny)
        result_one_batch_channel = regrid_icon_to_rotlatlon(data[1, 2], indices, weights, nx=nx, ny=ny)
        torch.testing.assert_close(result[1, 2], result_one_batch_channel)

    def test_result_clamped_to_stencil_range(self):
        """Result should be clamped between min and max of stencil values."""
        data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
        # Weights that would extrapolate outside [1, 3]
        weights = torch.tensor([[2.0, -0.5, -0.5]], dtype=torch.float32)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=1, ny=1)
        assert result.item() == pytest.approx(1.0)  # Should be clamped to min
        weights = torch.tensor([[-0.5, -0.5, 2.0]], dtype=torch.float32)
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=1, ny=1)
        assert result.item() == pytest.approx(3.0)  # Should be clamped to max

    def test_output_dtype_matches_input(self, simple_regrid_inputs):
        data, indices, weights = simple_regrid_inputs
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=3, ny=2)
        assert result.dtype == data.dtype

    def test_constant_data_gives_constant_output(self):
        """Constant input across the stencil should produce constant output."""
        val = 7.0
        data = torch.full((10,), val)
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        weights = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        result = regrid_icon_to_rotlatlon(data, indices, weights, nx=1, ny=2)
        torch.testing.assert_close(result, torch.full((2, 1), val))


############################################################################
#                          GridData — init                                 #
############################################################################


class TestGridDataInit:
    """Tests for GridData initialization and input validation."""

    @pytest.fixture
    def regular_grid_points(self):
        """Create a regular 5x5 grid of original points and target points inside it."""
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_orig = lons_orig.ravel()
        lats_orig = lats_orig.ravel()

        lons_target = np.array([1.0, 2.0, 3.0, 1.5])
        lats_target = np.array([1.0, 2.0, 3.0, 2.5])
        return lons_orig, lats_orig, lons_target, lats_target

    def test_creation(self, regular_grid_points):
        gd = GridData(*regular_grid_points)
        assert gd is not None

    def test_mismatched_orig_raises(self):
        with pytest.raises(ValueError, match="Original longitude and latitude"):
            GridData(
                np.array([0.0, 1.0]),
                np.array([0.0]),
                np.array([0.5]),
                np.array([0.5]),
            )

    def test_mismatched_target_raises(self):
        with pytest.raises(ValueError, match="Target longitude and latitude"):
            GridData(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([0.5, 0.6]),
                np.array([0.5]),
            )

    def test_initial_state_is_numpy(self, regular_grid_points):
        gd = GridData(*regular_grid_points)
        assert gd.is_torch is False
        assert gd.device is None
        assert isinstance(gd._lambda1, np.ndarray)
        assert isinstance(gd._lambda2, np.ndarray)
        assert isinstance(gd._lambda3, np.ndarray)
        assert isinstance(gd._simplex_id, np.ndarray)
        assert isinstance(gd._tri.simplices, np.ndarray)

    def test_barycentric_weights_sum_to_one(self, regular_grid_points):
        gd = GridData(*regular_grid_points)
        total = gd._lambda1 + gd._lambda2 + gd._lambda3
        np.testing.assert_allclose(total, 1.0, atol=1e-12)

    def test_barycentric_weights_non_negative_for_interior(self, regular_grid_points):
        """Points inside the convex hull should have non-negative barycentric coords."""
        gd = GridData(*regular_grid_points)
        inside = gd._simplex_id != -1
        assert np.all(gd._lambda1[inside] >= -1e-12)
        assert np.all(gd._lambda2[inside] >= -1e-12)
        assert np.all(gd._lambda3[inside] >= -1e-12)


############################################################################
#                      GridData — to_torch / to_numpy                      #
############################################################################


class TestGridDataDeviceConversion:
    """Tests for to_torch and to_numpy conversions."""

    @pytest.fixture
    def grid_data(self):
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_target = np.array([1.0, 2.0, 3.0])
        lats_target = np.array([1.0, 2.0, 3.0])
        return GridData(
            lons_orig.ravel(), lats_orig.ravel(), lons_target, lats_target
        )

    def test_to_torch_sets_flag(self, grid_data):
        grid_data.to_torch()
        assert grid_data.is_torch is True
        assert grid_data.device == torch.device("cpu")

    def test_to_torch_produces_tensors(self, grid_data):
        grid_data.to_torch()
        assert isinstance(grid_data._lambda1, torch.Tensor)
        assert isinstance(grid_data._lambda2, torch.Tensor)
        assert isinstance(grid_data._lambda3, torch.Tensor)
        assert isinstance(grid_data._simplex_id, torch.Tensor)
        assert isinstance(grid_data._tri.simplices, torch.Tensor)

    def test_to_torch_string_device(self, grid_data):
        grid_data.to_torch("cpu")
        assert grid_data.device == torch.device("cpu")
        assert grid_data.is_torch is True

    def test_to_numpy_restores_arrays(self, grid_data):
        grid_data.to_torch()
        grid_data.to_numpy()
        assert grid_data.is_torch is False
        assert grid_data.device is None
        assert isinstance(grid_data._lambda1, np.ndarray)
        assert isinstance(grid_data._lambda2, np.ndarray)
        assert isinstance(grid_data._lambda3, np.ndarray)
        assert isinstance(grid_data._simplex_id, np.ndarray)
        assert isinstance(grid_data._tri.simplices, np.ndarray)

    def test_to_numpy_noop_when_already_numpy(self, grid_data):
        """Calling to_numpy when already numpy should be a no-op."""
        lambda1_before = grid_data._lambda1
        grid_data.to_numpy()
        assert grid_data._lambda1 is lambda1_before

    def test_roundtrip_preserves_values(self, grid_data):
        lambda1_orig = grid_data._lambda1.copy()
        lambda2_orig = grid_data._lambda2.copy()
        lambda3_orig = grid_data._lambda3.copy()
        simplex_id_orig = grid_data._simplex_id.copy()
        tri_simplices_orig = grid_data._tri.simplices.copy()

        grid_data.to_torch()
        grid_data.to_numpy()

        np.testing.assert_allclose(grid_data._lambda1, lambda1_orig, atol=1e-12)
        np.testing.assert_allclose(grid_data._lambda2, lambda2_orig, atol=1e-12)
        np.testing.assert_allclose(grid_data._lambda3, lambda3_orig, atol=1e-12)
        np.testing.assert_array_equal(grid_data._simplex_id, simplex_id_orig)
        np.testing.assert_array_equal(grid_data._tri.simplices, tri_simplices_orig)


############################################################################
#                     GridData — interpolate (numpy)                       #
############################################################################


class TestGridDataInterpolateNumpy:
    """Tests for the interpolate method using numpy arrays."""

    @pytest.fixture
    def grid_data_on_regular(self):
        """GridData with a regular grid as source and a few interior targets."""
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_target = np.array([1.0, 2.0, 3.0])
        lats_target = np.array([1.0, 2.0, 3.0])
        return GridData(
            lons_orig.ravel(), lats_orig.ravel(), lons_target, lats_target
        )

    def test_interpolate_constant_field(self, grid_data_on_regular):
        """A constant field should interpolate to the same constant."""
        n_orig = len(grid_data_on_regular.longitudes_orig)
        values = np.full((1, n_orig), 5.0)
        result = grid_data_on_regular.interpolate(values)
        np.testing.assert_allclose(result, 5.0, atol=1e-12)

    def test_interpolate_linear_field(self, grid_data_on_regular):
        """A linear field f(x,y) = x + y should be reproduced exactly."""
        gd = grid_data_on_regular
        lons = gd.longitudes_orig
        lats = gd.latitudes_orig
        values = (lons + lats)[np.newaxis, :]  # (1, n_orig)

        result = gd.interpolate(values)  # (1, n_target)
        expected = gd.longitudes_target + gd.latitudes_target
        np.testing.assert_allclose(result[0], expected, atol=1e-10)

    def test_interpolate_multichannel(self, grid_data_on_regular):
        """Multiple channels should be interpolated independently."""
        gd = grid_data_on_regular
        n_orig = len(gd.longitudes_orig)
        n_channels = 3
        values = np.random.RandomState(42).randn(n_channels, n_orig)
        result = gd.interpolate(values)
        assert result.shape == (n_channels, len(gd.longitudes_target))

    def test_interpolate_batch_and_channels(self, grid_data_on_regular):
        """Batch + channel dimensions should be preserved."""
        gd = grid_data_on_regular
        n_orig = len(gd.longitudes_orig)
        batch, channels = 2, 3
        values = np.random.RandomState(0).randn(batch, channels, n_orig)
        result = gd.interpolate(values)
        assert result.shape == (batch, channels, len(gd.longitudes_target))

    def test_interpolate_wrong_last_dim_raises(self, grid_data_on_regular):
        with pytest.raises(ValueError, match="Expected values with shape"):
            grid_data_on_regular.interpolate(np.zeros((1, 7)))

    def test_fill_value_for_outside_points(self):
        """Points outside the convex hull should get fill_value."""
        lons_orig = np.array([0.0, 1.0, 0.0, 1.0])
        lats_orig = np.array([0.0, 0.0, 1.0, 1.0])
        # One inside, one far outside
        lons_target = np.array([0.5, 10.0])
        lats_target = np.array([0.5, 10.0])

        gd = GridData(lons_orig, lats_orig, lons_target, lats_target)
        values = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = gd.interpolate(values, fill_value=-999.0)

        # Outside point should be fill_value
        assert result[0, 1] == -999.0

    def test_fill_value_default_nan(self):
        """Default fill_value should be NaN."""
        lons_orig = np.array([0.0, 1.0, 0.0, 1.0])
        lats_orig = np.array([0.0, 0.0, 1.0, 1.0])
        lons_target = np.array([0.5, 10.0])
        lats_target = np.array([0.5, 10.0])

        gd = GridData(lons_orig, lats_orig, lons_target, lats_target)
        values = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = gd.interpolate(values)
        assert np.isnan(result[0, 1])

    def test_callable_alias(self, grid_data_on_regular):
        """__call__ should produce the same result as interpolate."""
        gd = grid_data_on_regular
        n_orig = len(gd.longitudes_orig)
        values = np.random.RandomState(7).randn(1, n_orig)
        np.testing.assert_array_equal(gd(values), gd.interpolate(values))

    def test_channel_batch_consistency(self, grid_data_on_regular):
        """Interpolate should give consistent results across channels and batches."""
        gd = grid_data_on_regular
        n_orig = len(gd.longitudes_orig)
        batch, channels = 2, 3
        values = np.random.RandomState(123).randn(batch, channels, n_orig)
        result = gd.interpolate(values)
        result_one_batch_channel = gd.interpolate(values[1, 1])

        np.testing.assert_allclose(result[1, 1], result_one_batch_channel, atol=1e-12)


############################################################################
#                     GridData — interpolate (torch)                       #
############################################################################


class TestGridDataInterpolateTorch:
    """Tests for interpolation using PyTorch tensors."""

    @pytest.fixture
    def grid_data_torch(self):
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_target = np.array([1.0, 2.0, 3.0])
        lats_target = np.array([1.0, 2.0, 3.0])
        gd = GridData(
            lons_orig.ravel(), lats_orig.ravel(), lons_target, lats_target
        )
        gd.to_torch()
        return gd

    def test_interpolate_constant_field_torch(self, grid_data_torch):
        gd = grid_data_torch
        n_orig = len(gd.longitudes_orig)
        values = torch.full((1, n_orig), 5.0)
        result = gd.interpolate(values)
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.full_like(result, 5.0), atol=1e-6, rtol=0)

    def test_interpolate_linear_field_torch(self, grid_data_torch):
        gd = grid_data_torch
        lons = torch.from_numpy(gd.longitudes_orig)
        lats = torch.from_numpy(gd.latitudes_orig)
        values = (lons + lats).unsqueeze(0)  # (1, n_orig)
        result = gd.interpolate(values)  # (1, n_target)
        expected = torch.from_numpy(gd.longitudes_target + gd.latitudes_target)
        torch.testing.assert_close(result[0], expected, atol=1e-6, rtol=1e-5)

    def test_interpolate_multichannel_torch(self, grid_data_torch):
        gd = grid_data_torch
        n_orig = len(gd.longitudes_orig)
        n_channels = 3
        values = torch.randn(n_channels, n_orig)
        result = gd.interpolate(values)
        assert result.shape == (n_channels, len(gd.longitudes_target))

    def test_interpolate_batch_and_channels_torch(self, grid_data_torch):
        gd = grid_data_torch
        n_orig = len(gd.longitudes_orig)
        batch, channels = 2, 3
        values = torch.randn(batch, channels, n_orig)
        result = gd.interpolate(values)
        assert result.shape == (batch, channels, len(gd.longitudes_target))

    def test_interpolate_wrong_last_dim_raises_torch(self, grid_data_torch):
        with pytest.raises(ValueError, match="Expected values with shape"):
            grid_data_torch.interpolate(torch.zeros((1, 7)))

    def test_fill_value_for_outside_points_torch(self):
        lons_orig = np.array([0.0, 1.0, 0.0, 1.0])
        lats_orig = np.array([0.0, 0.0, 1.0, 1.0])
        lons_target = np.array([0.5, 10.0])
        lats_target = np.array([0.5, 10.0])

        gd = GridData(lons_orig, lats_orig, lons_target, lats_target)
        gd.to_torch()
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = gd.interpolate(values, fill_value=-999.0)
        assert result[0, 1].item() == -999.0


    def test_fill_value_default_nan_torch(self):
        lons_orig = np.array([0.0, 1.0, 0.0, 1.0])
        lats_orig = np.array([0.0, 0.0, 1.0, 1.0])
        lons_target = np.array([0.5, 10.0])
        lats_target = np.array([0.5, 10.0])

        gd = GridData(lons_orig, lats_orig, lons_target, lats_target)
        gd.to_torch()
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = gd.interpolate(values)
        assert torch.isnan(result[0, 1])

    def test_channel_batch_consistency(self, grid_data_torch):
        gd = grid_data_torch
        n_orig = len(gd.longitudes_orig)
        batch, channels = 2, 3
        values = torch.randn(batch, channels, n_orig)
        result = gd.interpolate(values)
        result_one_batch_channel = gd.interpolate(values[1, 1])
        torch.testing.assert_close(result[1, 1], result_one_batch_channel, atol=1e-6, rtol=0)

    def test_numpy_and_torch_agree(self):
        """Numpy and Torch paths should produce the same results."""
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_target = np.array([1.0, 2.5, 3.5])
        lats_target = np.array([1.0, 2.5, 3.5])

        gd_np = GridData(
            lons_orig.ravel(), lats_orig.ravel(), lons_target, lats_target
        )
        rng = np.random.RandomState(99)
        values_np = rng.randn(2, len(lons_orig.ravel()))

        result_np = gd_np.interpolate(values_np)

        gd_np.to_torch()
        values_torch = torch.from_numpy(values_np)
        result_torch = gd_np.interpolate(values_torch)

        np.testing.assert_allclose(
            result_np, result_torch.numpy(), atol=1e-10
        )


############################################################################
#                    GridData — outside hull warning                       #
############################################################################


class TestGridDataOutsideHull:
    """Test behaviour when target points lie outside the convex hull."""

    def test_outside_hull_warning(self, capsys):
        lons_orig = np.array([0.0, 1.0, 0.0])
        lats_orig = np.array([0.0, 0.0, 1.0])
        lons_target = np.array([0.25, -5.0])
        lats_target = np.array([0.25, -5.0])

        GridData(lons_orig, lats_orig, lons_target, lats_target)
        captured = capsys.readouterr()
        assert "outside the convex hull" in captured.out

    def test_no_warning_when_all_inside(self, capsys):
        lons_orig, lats_orig = np.meshgrid(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5)
        )
        lons_target = np.array([1.0, 2.0])
        lats_target = np.array([1.0, 2.0])

        GridData(lons_orig.ravel(), lats_orig.ravel(), lons_target, lats_target)
        captured = capsys.readouterr()
        assert "outside the convex hull" not in captured.out
