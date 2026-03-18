import math

import pytest
import torch

from hirad.utils.patching import (
    BasePatching2D,
    GridPatching2D,
    RandomPatching2D,
    image_batching,
    image_fuse,
)


############################################################################
#                        BasePatching2D — init                             #
############################################################################


class TestBasePatching2DInit:
    """Tests for BasePatching2D initialization and validation."""

    def test_non_2d_img_shape_raises(self):
        """img_shape with wrong number of dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="img_shape must be 2D"):
            # Use GridPatching2D as a concrete subclass
            GridPatching2D(img_shape=(64, 64, 3), patch_shape=(32, 32))

    def test_non_2d_patch_shape_raises(self):
        """patch_shape with wrong number of dimensions should raise ValueError."""
        with pytest.raises(ValueError, match="patch_shape must be 2D"):
            GridPatching2D(img_shape=(64, 64), patch_shape=(32, 32, 1))

    def test_patch_larger_than_image_warns(self):
        """patch_shape larger than img_shape should issue a warning."""
        with pytest.warns(UserWarning, match="larger than"):
            GridPatching2D(img_shape=(32, 32), patch_shape=(64, 64))

    def test_patch_clamped_to_image_shape(self):
        """patch_shape should be clamped to img_shape when it exceeds it."""
        with pytest.warns(UserWarning):
            patcher = GridPatching2D(img_shape=(32, 48), patch_shape=(64, 64))
        assert patcher.patch_shape == (32, 48)

    def test_valid_shapes_stored(self):
        patcher = GridPatching2D(img_shape=(64, 128), patch_shape=(32, 32))
        assert patcher.img_shape == (64, 128)
        assert patcher.patch_shape == (32, 32)


############################################################################
#                     BasePatching2D — global_index                        #
############################################################################


class TestBasePatching2DGlobalIndex:
    """Tests for the global_index method."""

    def test_global_index_shape(self):
        patcher = GridPatching2D(img_shape=(64, 64), patch_shape=(32, 32))
        gi = patcher.global_index(batch_size=1)
        assert gi.ndim == 4
        assert gi.shape[1] == 2
        assert gi.shape[2] == patcher.patch_shape[0]
        assert gi.shape[3] == patcher.patch_shape[1]

    def test_global_index_values_within_image(self):
        """All global indices should fall within the original image dimensions."""
        patcher = GridPatching2D(
            img_shape=(64, 128), patch_shape=(32, 32), overlap_pix=4
        )
        gi = patcher.global_index(batch_size=1)
        assert gi[:, 0].min() >= 0
        assert gi[:, 1].min() >= 0
        # Padded indices may exceed image shape, but y/x coords should be valid
        assert gi[:, 0].max() < patcher.img_shape[0] + patcher.patch_shape[0]
        assert gi[:, 1].max() < patcher.img_shape[1] + patcher.patch_shape[1]

    def test_global_index_values_simple_case(self):
        """ In a simple 4x4 image with 2x2 patches and no overlap, global indices should be predictable. """
        patcher = GridPatching2D(img_shape=(4, 4), patch_shape=(2, 2), overlap_pix=0)
        gi = patcher.global_index(batch_size=1)
        expected_indices = torch.tensor([
            [[[0, 0], [1, 1]], [[0, 1], [0, 1]]],
            [[[2, 2], [3, 3]], [[0, 1], [0, 1]]],
            [[[0, 0], [1, 1]], [[2, 3], [2, 3]]],
            [[[2, 2], [3, 3]], [[2, 3], [2, 3]]]
        ])
        assert torch.equal(gi.cpu(), expected_indices)

    def test_global_index_device(self):
        patcher = GridPatching2D(img_shape=(32, 32), patch_shape=(16, 16))
        gi = patcher.global_index(batch_size=1, device="cpu")
        assert gi.device == torch.device("cpu")


############################################################################
#                     BasePatching2D — fuse not implemented                #
############################################################################


class TestBasePatching2DFuse:
    """Tests that fuse raises NotImplementedError for subclasses that don't implement it."""

    def test_random_patching_fuse_raises(self):
        """RandomPatching2D does not implement fuse."""
        patcher = RandomPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), patch_num=4
        )
        dummy = torch.randn(4, 3, 32, 32)
        with pytest.raises(NotImplementedError, match="fuse"):
            patcher.fuse(dummy)


############################################################################
#                   BasePatching2D — apply abstract method                 #
############################################################################


class TestBasePatching2DApply:
    """Tests that apply raises NotImplementedError for subclasses that don't implement it."""

    def test_grid_patching_apply_raises(self):
        """BasePatching2D does not implement apply."""
        with pytest.raises(TypeError, match="apply"):
            patcher = BasePatching2D(img_shape=(64, 64), patch_shape=(32, 32))


############################################################################
#                     RandomPatching2D — init                              #
############################################################################


class TestRandomPatching2DInit:
    """Tests for RandomPatching2D initialization."""

    def test_patch_num_stored(self):
        patcher = RandomPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), patch_num=8
        )
        assert patcher.patch_num == 8

    def test_patch_indices_generated_on_init(self):
        patcher = RandomPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), patch_num=5
        )
        assert len(patcher.patch_indices) == 5

    def test_patch_indices_within_bounds(self):
        img_h, img_w = 100, 120
        patch_h, patch_w = 30, 40
        patcher = RandomPatching2D(
            img_shape=(img_h, img_w), patch_shape=(patch_h, patch_w), patch_num=20
        )
        for py, px in patcher.patch_indices:
            assert 0 <= py <= img_h - patch_h
            assert 0 <= px <= img_w - patch_w


############################################################################
#                 RandomPatching2D — set / reset indices                   #
############################################################################


class TestRandomPatching2DIndices:
    """Tests for patch index manipulation."""

    def test_reset_changes_indices(self):
        """Resetting should produce new random indices (with overwhelming probability)."""
        patcher = RandomPatching2D(
            img_shape=(256, 256), patch_shape=(32, 32), patch_num=50
        )
        old_indices = list(patcher.patch_indices)
        patcher.reset_patch_indices()
        # Extremely unlikely to be identical for 50 patches on a 256x256 image
        assert patcher.patch_indices != old_indices

    def test_get_patch_indices_returns_current(self):
        patcher = RandomPatching2D(
            img_shape=(64, 64), patch_shape=(16, 16), patch_num=3
        )
        assert patcher.get_patch_indices() is patcher.patch_indices

    def test_set_patch_num_updates_count_and_indices(self):
        patcher = RandomPatching2D(
            img_shape=(64, 64), patch_shape=(16, 16), patch_num=3
        )
        patcher.set_patch_num(10)
        assert patcher.patch_num == 10
        assert len(patcher.patch_indices) == 10


############################################################################
#                      RandomPatching2D — apply                            #
############################################################################


class TestRandomPatching2DApply:
    """Tests for RandomPatching2D.apply."""

    @pytest.fixture
    def patcher_and_input(self):
        img_shape = (64, 64)
        patch_shape = (16, 16)
        patch_num = 4
        patcher = RandomPatching2D(img_shape, patch_shape, patch_num)
        batch_size, channels = 2, 3
        x = torch.randn(batch_size, channels, *img_shape)
        return patcher, x, batch_size, channels

    def test_output_shape(self, patcher_and_input):
        patcher, x, batch_size, channels = patcher_and_input
        out = patcher.apply(x)
        assert out.shape == (
            batch_size * patcher.patch_num,
            channels,
            patcher.patch_shape[0],
            patcher.patch_shape[1],
        )

    def test_output_values_match_input_slices(self, patcher_and_input):
        """Each patch in the output should correspond to the correct slice of input."""
        patcher, x, batch_size, _ = patcher_and_input
        out = patcher.apply(x)
        for i, (py, px) in enumerate(patcher.patch_indices):
            expected = x[
                :, :,
                py : py + patcher.patch_shape[0],
                px : px + patcher.patch_shape[1],
            ]
            torch.testing.assert_close(
                out[batch_size * i : batch_size * (i + 1)], expected
            )

    def test_apply_with_additional_input(self, patcher_and_input):
        """Additional input should be concatenated along channel dim."""
        patcher, x, batch_size, channels = patcher_and_input
        add_channels = 2
        additional = torch.randn(batch_size, add_channels, 32, 32)
        out = patcher.apply(x, additional_input=additional)
        assert out.shape[1] == channels + add_channels
        assert torch.allclose(out[:, :channels], patcher.apply(x))
        assert torch.allclose(out[:, channels:], torch.nn.functional.interpolate(
            additional, size=patcher.patch_shape, mode="bilinear").repeat(patcher.patch_num, 1, 1, 1))

    def test_apply_single_patch(self):
        """Test with a single patch."""
        patcher = RandomPatching2D(
            img_shape=(32, 32), patch_shape=(32, 32), patch_num=1
        )
        x = torch.randn(1, 1, 32, 32)
        out = patcher.apply(x)
        torch.testing.assert_close(out, x)


############################################################################
#                      GridPatching2D — init                               #
############################################################################


class TestGridPatching2DInit:
    """Tests for GridPatching2D initialization."""

    def test_patch_num_no_overlap(self):
        """Without overlap, patches should tile the image exactly."""
        patcher = GridPatching2D(img_shape=(64, 64), patch_shape=(32, 32))
        expected_x = math.ceil(64 / 32)
        expected_y = math.ceil(64 / 32)
        assert patcher.patch_num == expected_x * expected_y

    def test_patch_num_with_overlap(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), overlap_pix=8
        )
        expected_x = math.ceil(64 / (32 - 8))
        expected_y = math.ceil(64 / (32 - 8))
        assert patcher.patch_num == expected_x * expected_y

    def test_patch_num_with_boundary(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), boundary_pix=4
        )
        expected_x = math.ceil(64 / (32 - 4))
        expected_y = math.ceil(64 / (32 - 4))
        assert patcher.patch_num == expected_x * expected_y

    def test_patch_num_with_overlap_and_boundary(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32),
            overlap_pix=8, boundary_pix=10
        )
        expected_x = math.ceil(64 / (32 - 8 - 10))
        expected_y = math.ceil(64 / (32 - 8 - 10))
        assert patcher.patch_num == expected_x * expected_y

    def test_non_divisible_image_shape(self):
        """Image dimensions that don't divide evenly by stride should still work."""
        patcher = GridPatching2D(img_shape=(100, 77), patch_shape=(32, 32))
        assert patcher.patch_num == math.ceil(100 / 32) * math.ceil(77 / 32)


############################################################################
#                      GridPatching2D — apply                              #
############################################################################


class TestGridPatching2DApply:
    """Tests for GridPatching2D.apply."""

    @pytest.fixture
    def grid_patcher_and_input(self):
        img_shape = (64, 64)
        patch_shape = (32, 32)
        patcher = GridPatching2D(img_shape, patch_shape)
        batch_size, channels = 2, 3
        x = torch.randn(batch_size, channels, *img_shape)
        return patcher, x, batch_size, channels

    def test_output_shape(self, grid_patcher_and_input):
        patcher, x, batch_size, channels = grid_patcher_and_input
        out = patcher.apply(x)
        assert out.shape == (
            batch_size * patcher.patch_num,
            channels,
            patcher.patch_shape[0],
            patcher.patch_shape[1],
        )

    def test_output_shape_with_additional_input(self, grid_patcher_and_input):
        patcher, x, batch_size, channels = grid_patcher_and_input
        add_channels = 5
        additional = torch.randn(batch_size, add_channels, 16, 16)
        out = patcher.apply(x, additional_input=additional)
        assert out.shape == (
            batch_size * patcher.patch_num,
            channels + add_channels,
            patcher.patch_shape[0],
            patcher.patch_shape[1],
        )

    def test_apply_with_overlap(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), overlap_pix=8
        )
        x = torch.randn(1, 1, 64, 64)
        out = patcher.apply(x)
        assert out.shape == (patcher.patch_num, 1, 32, 32)

    def test_apply_with_boundary(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), boundary_pix=4
        )
        x = torch.randn(1, 1, 64, 64)
        out = patcher.apply(x)
        assert out.shape == (patcher.patch_num, 1, 32, 32)

    def test_apply_with_overlap_and_boundary(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32),
            overlap_pix=8, boundary_pix=10
        )
        x = torch.randn(1, 1, 64, 64)
        out = patcher.apply(x)
        assert out.shape == (patcher.patch_num, 1, 32, 32)


############################################################################
#                      GridPatching2D — fuse                               #
############################################################################


class TestGridPatching2DFuse:
    """Tests for GridPatching2D.fuse."""

    def test_fuse_output_shape(self):
        img_shape = (64, 64)
        patcher = GridPatching2D(img_shape, patch_shape=(32, 32))
        batch_size, channels = 2, 3
        patches = torch.randn(
            batch_size * patcher.patch_num, channels,
            patcher.patch_shape[0], patcher.patch_shape[1],
        )
        fused = patcher.fuse(patches, batch_size=batch_size)
        assert fused.shape == (batch_size, channels, *img_shape)

    def test_fuse_output_shape_with_overlap(self):
        img_shape = (64, 128)
        patcher = GridPatching2D(
            img_shape, patch_shape=(32, 32), overlap_pix=8
        )
        batch_size, channels = 2, 2
        patches = torch.randn(
            batch_size * patcher.patch_num, channels,
            patcher.patch_shape[0], patcher.patch_shape[1],
        )
        fused = patcher.fuse(patches, batch_size=batch_size)
        assert fused.shape == (batch_size, channels, *img_shape)


############################################################################
#                 GridPatching2D — roundtrip (apply → fuse)                #
############################################################################


class TestGridPatching2DRoundtrip:
    """Tests that apply followed by fuse reconstructs the original image."""

    @pytest.mark.parametrize(
        "img_shape, patch_shape, overlap_pix, boundary_pix",
        [
            ((64, 64), (32, 32), 0, 0),
            ((64, 64), (32, 32), 8, 0),
            ((64, 64), (32, 32), 0, 4),
            ((64, 64), (32, 32), 8, 4),
            ((100, 77), (32, 32), 4, 2),
            ((48, 96), (24, 48), 6, 0),
        ],
    )
    def test_roundtrip_reconstructs_image(
        self, img_shape, patch_shape, overlap_pix, boundary_pix
    ):
        """Patching and then fusing should recover the original image."""
        patcher = GridPatching2D(
            img_shape, patch_shape,
            overlap_pix=overlap_pix, boundary_pix=boundary_pix,
        )
        batch_size, channels = 2, 3
        x = torch.randn(batch_size, channels, *img_shape)
        patches = patcher.apply(x)
        reconstructed = patcher.fuse(patches, batch_size=batch_size)
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_roundtrip_single_patch_covers_image(self):
        """A single patch covering the full image should roundtrip exactly."""
        img_shape = (32, 32)
        patcher = GridPatching2D(img_shape, patch_shape=(32, 32))
        x = torch.randn(1, 1, *img_shape)
        patches = patcher.apply(x)
        reconstructed = patcher.fuse(patches, batch_size=1)
        torch.testing.assert_close(reconstructed, x)

    def test_roundtrip_preserves_dtype(self):
        patcher = GridPatching2D(
            img_shape=(64, 64), patch_shape=(32, 32), overlap_pix=4
        )
        x = torch.randn(1, 1, 64, 64, dtype=torch.float64)
        patches = patcher.apply(x)
        reconstructed = patcher.fuse(patches, batch_size=1)
        assert reconstructed.dtype == x.dtype


############################################################################
#                        image_batching — function                         #
############################################################################


class TestImageBatching:
    """Tests for the image_batching standalone function."""

    def test_output_shape_no_overlap(self):
        x = torch.randn(2, 3, 64, 64)
        out = image_batching(x, patch_shape_y=32, patch_shape_x=32,
                             overlap_pix=0, boundary_pix=0)
        patch_num = math.ceil(64 / 32) * math.ceil(64 / 32)
        assert out.shape == (patch_num * 2, 3, 32, 32)

    def test_output_shape_with_interp(self):
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        interp = torch.randn(batch_size, 5, 32, 32)
        out = image_batching(x, 32, 32, overlap_pix=0, boundary_pix=0,
                             input_interp=interp)
        assert out.shape == (math.ceil(64 / 32) * math.ceil(64 / 32) * batch_size, 3 + 5, 32, 32)

    def test_invalid_patch_shape_x_raises(self):
        x = torch.randn(1, 1, 64, 64)
        with pytest.raises(ValueError, match="patch_shape_x"):
            image_batching(x, patch_shape_y=32, patch_shape_x=2,
                           overlap_pix=1, boundary_pix=1)

    def test_invalid_patch_shape_y_raises(self):
        x = torch.randn(1, 1, 64, 64)
        with pytest.raises(ValueError, match="patch_shape_y"):
            image_batching(x, patch_shape_y=2, patch_shape_x=32,
                           overlap_pix=1, boundary_pix=1)

    def test_interp_batch_mismatch_raises(self):
        x = torch.randn(2, 3, 64, 64)
        interp = torch.randn(3, 5, 32, 32)  # wrong batch size
        with pytest.raises(ValueError, match="batch size"):
            image_batching(x, 32, 32, 0, 0, input_interp=interp)

    def test_interp_shape_mismatch_raises(self):
        x = torch.randn(2, 3, 64, 64)
        interp = torch.randn(2, 5, 16, 16)  # wrong spatial dims
        with pytest.raises(ValueError, match="patch shape"):
            image_batching(x, 32, 32, 0, 0, input_interp=interp)

    def test_patch_too_small_for_overlap_and_boundary_x_raises(self):
        x = torch.randn(1, 1, 64, 64)
        with pytest.raises(ValueError, match="patch_shape_x"):
            image_batching(x, 32, 11, overlap_pix=5, boundary_pix=3)

    def test_patch_too_small_for_overlap_and_boundary_y_raises(self):
        x = torch.randn(1, 1, 64, 64)
        with pytest.raises(ValueError, match="patch_shape_y"):
            image_batching(x, 11, 32, overlap_pix=5, boundary_pix=3)

    def test_int32_input_preserves_dtype(self):
        x = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
        out = image_batching(x, 16, 16, 0, 0)
        assert out.dtype == torch.int32

    def test_int64_input_preserves_dtype(self):
        x = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int64)
        out = image_batching(x, 16, 16, 0, 0)
        assert out.dtype == torch.int64

    def test_patch_is_matching_the_original(self):
        """Patches should match the corresponding slices of the original image."""
        x = torch.randn(3, 2, 32, 32)
        patches = image_batching(x, 16, 16, overlap_pix=0, boundary_pix=0)
        expected_patches = torch.cat([
            x[:, :, 0:16, 0:16],
            x[:, :, 16:32, 0:16],
            x[:, :, 0:16, 16:32],
            x[:, :, 16:32, 16:32],
        ], dim=0)
        torch.testing.assert_close(patches, expected_patches)

    def test_patch_is_matching_the_original_with_overlap_and_boundary(self):
        """Patches should match the corresponding slices of the original image, even with overlap and boundary."""
        x = torch.randn(3, 2, 32, 32)
        patches = image_batching(x, 16, 16, overlap_pix=4, boundary_pix=2)
        # test if the patches at the corners and center match the expected slices of the original image
        # where padding is applied, compare only to the valid region of the original image
        # padding can be changed without affecting the validity of the extracted patch region, so we focus on the original image slices
        expected_patch_middle = x[:, :, 8:24, 8:24]
        expected_patch_top_left = x[:, :, 0:14, 0:14]
        expected_patch_bottom_left = x[:, :, 28:, 0:14]
        expected_patch_top_right = x[:, :, 0:14, 28:]
        expected_patch_bottom_right = x[:, :, 28:, 28:]
        torch.testing.assert_close(patches[3*5:3*6], expected_patch_middle)
        torch.testing.assert_close(patches[0:3,:,2:,2:], expected_patch_top_left)
        torch.testing.assert_close(patches[3*3:3*4,:,:4,2:], expected_patch_bottom_left)
        torch.testing.assert_close(patches[3*12:3*13, :, 2:, :4], expected_patch_top_right)
        torch.testing.assert_close(patches[3*15:, :, :4, :4], expected_patch_bottom_right)

############################################################################
#                          image_fuse — function                           #
############################################################################


class TestImageFuse:
    """Tests for the image_fuse standalone function."""

    def test_output_shape(self):
        batch_size = 2
        img_shape_y, img_shape_x = 64, 64
        patch_shape_y, patch_shape_x = 32, 32
        patch_num_x = math.ceil(img_shape_x / patch_shape_x)
        patch_num_y = math.ceil(img_shape_y / patch_shape_y)
        patch_num = patch_num_x * patch_num_y
        channels = 3
        patches = torch.randn(patch_num * batch_size, channels,
                               patch_shape_y, patch_shape_x)
        out = image_fuse(patches, img_shape_y, img_shape_x,
                         batch_size, overlap_pix=0, boundary_pix=0)
        assert out.shape == (batch_size, channels, img_shape_y, img_shape_x)

    def test_fuse_constant_patches(self):
        """Fusing constant-valued patches should yield a constant image."""
        val = 5.0
        img_shape = (32, 32)
        patcher = GridPatching2D(img_shape, patch_shape=(16, 16))
        patches = torch.full(
            (patcher.patch_num, 1, 16, 16), val
        )
        fused = image_fuse(patches, img_shape[0], img_shape[1],
                           batch_size=1, overlap_pix=0, boundary_pix=0)
        torch.testing.assert_close(fused, torch.full((1, 1, *img_shape), val))

    #TODO: after normalizing by overlap count, the output may not be exactly the same as the input for integer types, so we would need to round and cast back to the original dtype. 
    # We can add it after implementing that logic in image_fuse, but first we have to see if it would affect existing model checkpoints.
    # def test_int32_dtype_preserved(self):
    #     x = torch.randint(0, 100, (1, 1, 8, 8), dtype=torch.int32)
    #     patches = image_batching(x, 4, 4, 0, 0)
    #     fused = image_fuse(patches, 8, 8, batch_size=1,
    #                        overlap_pix=0, boundary_pix=0)
    #     print(fused)
    #     assert fused.dtype == torch.int32

    # def test_int64_dtype_preserved(self):
    #     x = torch.randint(0, 100, (1, 1, 8, 8), dtype=torch.int64)
    #     patches = image_batching(x, 4, 4, 0, 0)
    #     fused = image_fuse(patches, 8, 8, batch_size=1,
    #                        overlap_pix=0, boundary_pix=0)
    #     assert fused.dtype == torch.int64


############################################################################
#              image_batching + image_fuse — roundtrip                     #
############################################################################


class TestImageBatchingFuseRoundtrip:
    """Tests that image_batching followed by image_fuse recovers the original."""

    @pytest.mark.parametrize(
        "img_shape_y, img_shape_x, patch_shape_y, patch_shape_x, overlap_pix, boundary_pix",
        [
            (64, 64, 32, 32, 0, 0),
            (64, 64, 32, 32, 8, 0),
            (64, 64, 32, 32, 0, 4),
            (64, 64, 32, 32, 8, 4),
            (48, 96, 24, 48, 0, 0),
            (100, 77, 32, 32, 4, 2),
        ],
    )
    def test_roundtrip(
        self, img_shape_y, img_shape_x,
        patch_shape_y, patch_shape_x,
        overlap_pix, boundary_pix,
    ):
        batch_size, channels = 2, 3
        x = torch.randn(batch_size, channels, img_shape_y, img_shape_x)
        patches = image_batching(
            x, patch_shape_y, patch_shape_x, overlap_pix, boundary_pix
        )
        reconstructed = image_fuse(
            patches, img_shape_y, img_shape_x,
            batch_size, overlap_pix, boundary_pix,
        )
        torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_roundtrip_channels_last(self):
        """Roundtrip should work with channels_last memory format."""
        x = torch.randn(2, 3, 64, 64).to(memory_format=torch.channels_last)
        patches = image_batching(x, 32, 32, 0, 0)
        reconstructed = image_fuse(patches, 64, 64, batch_size=2,
                                   overlap_pix=0, boundary_pix=0)
        torch.testing.assert_close(
            reconstructed.contiguous(), x.contiguous(), atol=1e-5, rtol=1e-5
        )
