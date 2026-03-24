# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn as nn

from hirad.models.song_unet import SongUNet, SongUNetPosEmbd


# ---------------------------------------------------------------------------
#  Helpers / fixtures — use small model configs for fast CPU tests
# ---------------------------------------------------------------------------

B = 2
IMG_RES = 32
IN_CH = 4
OUT_CH = 3
SMALL_CFG = dict(
    model_channels=32,
    channel_mult=[1, 2],
    num_blocks=1,
    attn_resolutions=[],
    dropout=0.0,
)


@pytest.fixture()
def small_unet():
    """Return a small SongUNet that runs on CPU."""
    return SongUNet(
        img_resolution=IMG_RES,
        in_channels=IN_CH,
        out_channels=OUT_CH,
        **SMALL_CFG,
    )


@pytest.fixture()
def noise_labels():
    return torch.randn(B)


@pytest.fixture()
def class_labels():
    return torch.randint(0, 2, (B, 1)).float()


@pytest.fixture()
def input_image():
    return torch.randn(B, IN_CH, IMG_RES, IMG_RES)


############################################################################
#                         SongUNet — __init__                              #
############################################################################


class TestSongUNetInitValidation:
    """Test __init__ input validation."""

    def test_invalid_embedding_type_raises(self):
        with pytest.raises(ValueError, match="Invalid embedding_type"):
            SongUNet(
                img_resolution=IMG_RES,
                in_channels=IN_CH,
                out_channels=OUT_CH,
                embedding_type="invalid",
                **SMALL_CFG,
            )

    def test_invalid_encoder_type_raises(self):
        with pytest.raises(ValueError, match="Invalid encoder_type"):
            SongUNet(
                img_resolution=IMG_RES,
                in_channels=IN_CH,
                out_channels=OUT_CH,
                encoder_type="invalid",
                **SMALL_CFG,
            )

    def test_invalid_decoder_type_raises(self):
        with pytest.raises(ValueError, match="Invalid decoder_type"):
            SongUNet(
                img_resolution=IMG_RES,
                in_channels=IN_CH,
                out_channels=OUT_CH,
                decoder_type="invalid",
                **SMALL_CFG,
            )

    @pytest.mark.parametrize("etype", ["positional", "fourier", "zero"])
    def test_valid_embedding_types_accepted(self, etype):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type=etype,
            **SMALL_CFG,
        )
        assert model.embedding_type == etype

    @pytest.mark.parametrize("enc", ["standard", "skip", "residual"])
    def test_valid_encoder_types_accepted(self, enc):
        SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type=enc,
            **SMALL_CFG,
        )

    @pytest.mark.parametrize("dec", ["standard", "skip"])
    def test_valid_decoder_types_accepted(self, dec):
        SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            decoder_type=dec,
            **SMALL_CFG,
        )


class TestSongUNetInitResolution:
    """Test resolution handling in __init__."""

    def test_int_resolution_sets_square(self):
        model = SongUNet(
            img_resolution=32,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert model.img_shape_x == 32
        assert model.img_shape_y == 32

    def test_list_resolution_sets_height_width(self):
        model = SongUNet(
            img_resolution=[24, 32],
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert model.img_shape_y == 24
        assert model.img_shape_x == 32

    def test_img_resolution_stored(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert model.img_resolution == IMG_RES


class TestSongUNetInitEmbedding:
    """Test embedding-related initialization."""

    def test_positional_embedding_creates_map_noise(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="positional",
            **SMALL_CFG,
        )
        assert hasattr(model, "map_noise")
        assert hasattr(model, "map_layer0")
        assert hasattr(model, "map_layer1")

    def test_fourier_embedding_creates_map_noise(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="fourier",
            **SMALL_CFG,
        )
        assert hasattr(model, "map_noise")
        assert hasattr(model, "map_layer0")
        assert hasattr(model, "map_layer1")

    def test_zero_embedding_skips_mapping_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="zero",
            **SMALL_CFG,
        )
        assert not hasattr(model, "map_noise")
        assert not hasattr(model, "map_layer0")
        assert not hasattr(model, "map_layer1")
        assert not hasattr(model, "map_label")
        assert not hasattr(model, "map_augment")

    def test_emb_channels_computed_correctly(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            channel_mult_emb=4,
            **SMALL_CFG,
        )
        assert model.emb_channels == 32 * 4

    def test_label_dim_creates_map_label(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            label_dim=10,
            **SMALL_CFG,
        )
        assert model.map_label is not None

    def test_no_label_dim_map_label_is_none(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            label_dim=0,
            **SMALL_CFG,
        )
        assert model.map_label is None

    def test_augment_dim_creates_map_augment(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            augment_dim=5,
            **SMALL_CFG,
        )
        assert model.map_augment is not None

    def test_no_augment_dim_map_augment_is_none(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            augment_dim=0,
            **SMALL_CFG,
        )
        assert model.map_augment is None


class TestSongUNetInitEncoder:
    """Test encoder construction."""

    def test_encoder_module_dict_created(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert isinstance(model.enc, nn.ModuleDict)
        assert len(model.enc) > 0

    def test_skip_encoder_creates_aux_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type="skip",
            **SMALL_CFG,
        )
        aux_down_keys = [k for k in model.enc.keys() if "aux_down" in k]
        aux_skip_keys = [k for k in model.enc.keys() if "aux_skip" in k]
        assert len(aux_down_keys) > 0
        assert len(aux_skip_keys) > 0

    def test_residual_encoder_creates_aux_residual(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type="residual",
            **SMALL_CFG,
        )
        aux_keys = [k for k in model.enc.keys() if "aux_residual" in k]
        assert len(aux_keys) > 0

    def test_standard_encoder_has_no_aux_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type="standard",
            **SMALL_CFG,
        )
        aux_keys = [k for k in model.enc.keys() if "aux" in k]
        assert len(aux_keys) == 0

    def test_standard_encoder_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type="standard",
            **SMALL_CFG,
        )
        expected_layers = ["32x32_conv", "32x32_block0", "16x16_block0"]
        for layer in expected_layers:
            assert hasattr(model.enc, layer)
    


class TestSongUNetInitDecoder:
    """Test decoder construction."""

    def test_decoder_module_dict_created(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert isinstance(model.dec, nn.ModuleDict)
        assert len(model.dec) > 0

    def test_skip_decoder_creates_aux_up_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            decoder_type="skip",
            **SMALL_CFG,
        )
        aux_keys = [k for k in model.dec.keys() if "aux_up" in k]
        assert len(aux_keys) > 0

    def test_standard_decoder_has_no_aux_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            decoder_type="standard",
            **SMALL_CFG,
        )
        aux_keys = [k for k in model.dec.keys() if "aux_up" in k]
        assert len(aux_keys) == 0

    def test_standard_decoder_layers(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            decoder_type="standard",
            **SMALL_CFG,
        )
        expected_layers = ["16x16_in0", "16x16_in1", "16x16_block0", "16x16_block1", "32x32_up",
                           "32x32_block0", "32x32_block1", "32x32_aux_norm", "32x32_aux_conv"]
        for layer in expected_layers:
            assert hasattr(model.dec, layer)


class TestSongUNetInitAdditiveEmbed:
    """Test additive positional embedding in __init__."""

    def test_additive_pos_embed_creates_parameter(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            additive_pos_embed=True,
            **SMALL_CFG,
        )
        assert hasattr(model, "spatial_emb")
        assert isinstance(model.spatial_emb, nn.Parameter)
        assert model.spatial_emb.shape == (1, 32, IMG_RES, IMG_RES)

    def test_no_additive_pos_embed_by_default(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        assert not hasattr(model, "spatial_emb")


class TestSongUNetInitCheckpoint:
    """Test checkpoint level configuration."""

    def test_checkpoint_level_zero(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            checkpoint_level=0,
            **SMALL_CFG,
        )
        # threshold = (img_shape_y >> 0) + 1 = 32 >> 0 + 1 = 32 + 1 = 33
        assert model.checkpoint_threshold == IMG_RES + 1

    def test_checkpoint_level_one(self):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            checkpoint_level=1,
            **SMALL_CFG,
        )
        # threshold = (32 >> 1) + 1 = 16 + 1 = 17
        assert model.checkpoint_threshold == (IMG_RES >> 1) + 1


class TestSongUNetIsModule:
    """Test that SongUNet is a proper nn.Module."""

    def test_is_nn_module(self, small_unet):
        assert isinstance(small_unet, nn.Module)

    def test_has_parameters(self, small_unet):
        params = list(small_unet.parameters())
        assert len(params) > 0


############################################################################
#                        SongUNet — forward                                #
############################################################################


class TestSongUNetForwardShape:
    """Test forward pass output shapes."""

    def test_output_shape(self, small_unet, input_image, noise_labels, class_labels):
        out = small_unet(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_output_dtype_float32(self, small_unet, input_image, noise_labels, class_labels):
        out = small_unet(input_image, noise_labels, class_labels)
        assert out.dtype == torch.float32


class TestSongUNetForwardEmbeddingTypes:
    """Test forward with different embedding types."""

    def test_zero_embedding_forward(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="zero",
            **SMALL_CFG,
        )
        out = model(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_fourier_embedding_forward(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="fourier",
            **SMALL_CFG,
        )
        out = model(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_positional_embedding_forward(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="positional",
            **SMALL_CFG,
        )
        out = model(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)


class TestSongUNetForwardEncoderDecoder:
    """Test forward with various encoder/decoder combos."""

    @pytest.mark.parametrize("enc,dec", [
        ("standard", "standard"),
        ("skip", "standard"),
        ("residual", "standard"),
        ("standard", "skip"),
        ("skip", "skip"),
        ("residual", "skip"),
    ])
    def test_encoder_decoder_combinations(self, enc, dec, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            encoder_type=enc,
            decoder_type=dec,
            **SMALL_CFG,
        )
        out = model(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)


class TestSongUNetForwardLabel:
    """Test label dropout behavior during training."""

    def test_label_dropout_in_training(self, input_image, noise_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            label_dim=5,
            label_dropout=0.5,
            **SMALL_CFG,
        )
        model.train()
        labels = torch.ones(B, 5)
        out = model(input_image, noise_labels, labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_label_dropout_in_eval(self, input_image, noise_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            label_dim=5,
            label_dropout=0.5,
            **SMALL_CFG,
        )
        model.eval()
        labels = torch.ones(B, 5)
        out = model(input_image, noise_labels, labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_map_label_called_when_label_dim_positive(self, input_image, noise_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            label_dim=5,
            **SMALL_CFG,
        )
        assert model.map_label is not None
        called = []
        model.map_label.register_forward_hook(lambda m, i, o: called.append(True))
        out = model(input_image, noise_labels, torch.ones(B, 5))
        assert len(called) == 1


class TestSongUNetForwardAugment:
    """Test augment dropout behavior during training."""

    def test_map_augment_called_when_augment_dim_positive(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            augment_dim=3,
            **SMALL_CFG,
        )
        assert model.map_augment is not None
        called = []
        model.map_augment.register_forward_hook(lambda m, i, o: called.append(True))
        out = model(input_image, noise_labels, class_labels, augment_labels=torch.ones(B, 3))
        assert len(called) == 1

    def test_no_augment_labels_skips_map_augment(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            augment_dim=3,
            **SMALL_CFG,
        )
        assert model.map_augment is not None
        called = []
        model.map_augment.register_forward_hook(lambda m, i, o: called.append(True))
        out = model(input_image, noise_labels, class_labels)
        assert len(called) == 0


class TestSongUNetForwardNoiseEmbedding:
    def test_map_noise_called_when_embedding_type_fourier(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="fourier",
            **SMALL_CFG,
        )
        assert model.map_noise is not None
        called_map_noise = []
        called_map_layer0 = []
        called_map_layer1 = []
        model.map_noise.register_forward_hook(lambda m, i, o: called_map_noise.append(True))
        model.map_layer0.register_forward_hook(lambda m, i, o: called_map_layer0.append(True))
        model.map_layer1.register_forward_hook(lambda m, i, o: called_map_layer1.append(True))
        out = model(input_image, noise_labels, class_labels)
        assert len(called_map_noise) == 1
        assert len(called_map_layer0) == 1
        assert len(called_map_layer1) == 1

    def test_map_noise_called_when_embedding_type_positional(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            embedding_type="positional",
            **SMALL_CFG,
        )
        assert model.map_noise is not None
        called_map_noise = []
        called_map_layer0 = []
        called_map_layer1 = []
        model.map_noise.register_forward_hook(lambda m, i, o: called_map_noise.append(True))
        model.map_layer0.register_forward_hook(lambda m, i, o: called_map_layer0.append(True))
        model.map_layer1.register_forward_hook(lambda m, i, o: called_map_layer1.append(True))
        out = model(input_image, noise_labels, class_labels)
        assert len(called_map_noise) == 1
        assert len(called_map_layer0) == 1
        assert len(called_map_layer1) == 1


class TestSongUNetForwardEncoderDecoderCalls:
    """Test that all encoder and decoder blocks are called during forward."""

    def test_all_enc_dec_blocks_called_parametrized(
        self, input_image, noise_labels, class_labels
    ):
        """Test across different encoder/decoder combos."""
        for enc in ["standard", "skip", "residual"]:
            model = SongUNet(
                img_resolution=IMG_RES,
                in_channels=IN_CH,
                out_channels=OUT_CH,
                encoder_type=enc,
                **SMALL_CFG,
            )
            called = {}
            handles = []
            for name, block in list(model.enc.items()):
                called[name] = 0
                handle = block.register_forward_hook(
                    lambda m, i, o, n=name: called.__setitem__(n, called[n] + 1)
                )
                handles.append(handle)

            model(input_image, noise_labels, class_labels)

            for handle in handles:
                handle.remove()

            for name, count in called.items():
                assert count == 1, (
                    f"[enc={enc}] Block '{name}' called {count} times, expected 1"
                )

    def test_all_decoder_blocks_called_parametrized(
        self, input_image, noise_labels, class_labels
    ):
        """Test across different encoder/decoder combos."""
        for dec in ["standard", "skip"]:
            model = SongUNet(
                img_resolution=IMG_RES,
                in_channels=IN_CH,
                out_channels=OUT_CH,
                decoder_type=dec,
                **SMALL_CFG,
            )
            called = {}
            handles = []
            for name, block in list(model.dec.items()):
                called[name] = 0
                handle = block.register_forward_hook(
                    lambda m, i, o, n=name: called.__setitem__(n, called[n] + 1)
                )
                handles.append(handle)

            model(input_image, noise_labels, class_labels)

            for handle in handles:
                handle.remove()

            for name, count in called.items():
                assert count == 1, (
                    f"[dec={dec}] Block '{name}' called {count} times, expected 1"
                )


class TestSongUNetForwardAdditiveEmbed:
    """Test forward with additive positional embedding."""

    def test_additive_embed_forward(self, input_image, noise_labels, class_labels):
        model = SongUNet(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            additive_pos_embed=True,
            **SMALL_CFG,
        )
        out = model(input_image, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)


class TestSongUNetForwardRectangularResolution:
    """Test forward with non-square input."""

    def test_rectangular_resolution_forward(self, noise_labels, class_labels):
        res = [16, 32]
        model = SongUNet(
            img_resolution=res,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            **SMALL_CFG,
        )
        x = torch.randn(B, IN_CH, res[0], res[1])
        out = model(x, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, res[0], res[1])


############################################################################
#                  SongUNetPosEmbd — __init__                              #
############################################################################


# Positional embedding adds N_grid_channels to in_channels
N_GRID = 4
PE_IN_CH = IN_CH + N_GRID

PE_SMALL_CFG = dict(
    model_channels=32,
    channel_mult=[1, 2],
    num_blocks=1,
    attn_resolutions=[],
    dropout=0.0,
    use_apex_gn=False,
)


@pytest.fixture()
def small_pos_unet():
    return SongUNetPosEmbd(
        img_resolution=IMG_RES,
        in_channels=PE_IN_CH,
        out_channels=OUT_CH,
        N_grid_channels=N_GRID,
        **PE_SMALL_CFG,
    )


class TestSongUNetPosEmbdInitGridType:
    """Test grid type selection in __init__."""

    def test_sinusoidal_grid_default(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.gridtype == "sinusoidal"
        assert model.pos_embd.shape == (N_GRID, IMG_RES, IMG_RES)

    def test_learnable_grid(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="learnable",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.gridtype == "learnable"
        assert isinstance(model.pos_embd, nn.Parameter)
        assert model.pos_embd.shape == (N_GRID, IMG_RES, IMG_RES)

    def test_linear_grid(self):
        n_ch = 2
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + n_ch,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=n_ch,
            **PE_SMALL_CFG,
        )
        assert model.gridtype == "linear"
        assert model.pos_embd.shape == (2, IMG_RES, IMG_RES)

    def test_test_grid(self):
        n_ch = 2
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + n_ch,
            out_channels=OUT_CH,
            gridtype="test",
            N_grid_channels=n_ch,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (2, IMG_RES, IMG_RES)


class TestSongUNetPosEmbdInitGridChannelsValidation:
    """Test N_grid_channels validation."""

    def test_linear_grid_requires_2_channels(self):
        with pytest.raises(ValueError, match="N_grid_channels must be set to 2"):
            SongUNetPosEmbd(
                img_resolution=IMG_RES,
                in_channels=IN_CH + 4,
                out_channels=OUT_CH,
                gridtype="linear",
                N_grid_channels=4,
                **PE_SMALL_CFG,
            )

    def test_sinusoidal_multi_freq_requires_factor_of_4(self):
        with pytest.raises(ValueError, match="N_grid_channels must be a factor of 4"):
            SongUNetPosEmbd(
                img_resolution=IMG_RES,
                in_channels=IN_CH + 5,
                out_channels=OUT_CH,
                gridtype="sinusoidal",
                N_grid_channels=5,
                **PE_SMALL_CFG,
            )

    def test_sinusoidal_8_channels_accepted(self):
        n_ch = 8
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + n_ch,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=n_ch,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (n_ch, IMG_RES, IMG_RES)

    def test_zero_grid_channels_returns_none(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=0,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd is None

    def test_unsupported_gridtype_raises(self):
        with pytest.raises(ValueError, match="Gridtype not supported"):
            SongUNetPosEmbd(
                img_resolution=IMG_RES,
                in_channels=PE_IN_CH,
                out_channels=OUT_CH,
                gridtype="unknown",
                N_grid_channels=N_GRID,
                **PE_SMALL_CFG,
            )


class TestSongUNetPosEmbdInitLeadTime:
    """Test lead time related initialization."""

    def test_lead_time_mode_creates_lt_embd(self):
        lt_ch = 2
        lt_steps = 5
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + lt_ch,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=lt_ch,
            lead_time_steps=lt_steps,
            **PE_SMALL_CFG,
        )
        assert model.lead_time_mode is True
        assert model.lt_embd is not None
        assert model.lt_embd.shape == (lt_steps, lt_ch, IMG_RES, IMG_RES)

    def test_no_lead_time_mode_by_default(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.lead_time_mode is False

    def test_lead_time_none_channels_returns_none_embd(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=None,
            lead_time_steps=9,
            **PE_SMALL_CFG,
        )
        assert model.lt_embd is None

    def test_lead_time_none_steps_returns_none_embd(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=2,
            lead_time_steps=None,
            **PE_SMALL_CFG,
        )
        assert model.lt_embd is None


class TestSongUNetPosEmbdInitProbChannels:
    """Test prob_channels initialization."""

    def test_prob_channels_creates_scalar(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + 2,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=2,
            lead_time_steps=3,
            prob_channels=[0, 1],
            **PE_SMALL_CFG,
        )
        assert hasattr(model, "scalar")
        assert model.scalar.shape == (1, 2, 1, 1)

    def test_empty_prob_channels_no_scalar(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + 2,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=2,
            lead_time_steps=3,
            prob_channels=[],
            **PE_SMALL_CFG,
        )
        assert not hasattr(model, "scalar")


class TestSongUNetPosEmbdIsModule:
    """Test that SongUNetPosEmbd is a proper nn.Module and subclass of SongUNet."""

    def test_is_nn_module(self, small_pos_unet):
        assert isinstance(small_pos_unet, nn.Module)

    def test_is_subclass_of_song_unet(self, small_pos_unet):
        assert isinstance(small_pos_unet, SongUNet)


############################################################################
#              SongUNetPosEmbd — _get_positional_embedding                 #
############################################################################


class TestGetPositionalEmbedding:
    """Test _get_positional_embedding for various grid types."""

    def test_sinusoidal_4ch_grid_not_requires_grad(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert not model.pos_embd.requires_grad

    def test_linear_grid_not_requires_grad(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            **PE_SMALL_CFG,
        )
        assert not model.pos_embd.requires_grad

    def test_learnable_grid_requires_grad(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="learnable",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.requires_grad

    def test_sinusoidal_values_in_range(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.min() >= -1.0
        assert model.pos_embd.max() <= 1.0

    def test_linear_values_in_range(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.min() >= -1.0
        assert model.pos_embd.max() <= 1.0

    def test_rectangular_sinusoidal_grid(self):
        model = SongUNetPosEmbd(
            img_resolution=[16, 32],
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (N_GRID, 16, 32)

    def test_rectangular_grid_sinusoidal_8ch(self):
        n_ch = 8
        model = SongUNetPosEmbd(
            img_resolution=[16, 32],
            in_channels=IN_CH + n_ch,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=n_ch,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (n_ch, 16, 32)

    def test_rectangular_linear_grid(self):
        model = SongUNetPosEmbd(
            img_resolution=[16, 32],
            in_channels=IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (2, 16, 32)

    def test_rectangular_learnable_grid(self):
        model = SongUNetPosEmbd(
            img_resolution=[16, 32],
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="learnable",
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        assert model.pos_embd.shape == (N_GRID, 16, 32)

    def test_linear_grid_simple_values(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            **PE_SMALL_CFG,
        )
        # Check that the first channel is a vertical gradient and the second is horizontal
        for y in range(IMG_RES):
            for x in range(IMG_RES):
                expected_y = (y / (IMG_RES - 1)) * 2 - 1
                expected_x = (x / (IMG_RES - 1)) * 2 - 1
                assert torch.isclose(model.pos_embd[0, y, x], torch.tensor([expected_x]), atol=1e-5)
                assert torch.isclose(model.pos_embd[1, y, x], torch.tensor([expected_y]), atol=1e-5)

    def test_sinusoidal_grid_simple_values(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=4,
            **PE_SMALL_CFG,
        )
        # Check that the first two channels are sinusoids of different frequencies
        # and the next two channels are the cosine counterparts
        for y in range(IMG_RES):
            for x in range(IMG_RES):
                expected_ch0 = torch.sin(2 * torch.pi * torch.tensor([x]) / (IMG_RES - 1))
                expected_ch1 = torch.sin(2 * torch.pi * torch.tensor([y]) / (IMG_RES - 1))
                expected_ch2 = torch.cos(2 * torch.pi * torch.tensor([x]) / (IMG_RES - 1))
                expected_ch3 = torch.cos(2 * torch.pi * torch.tensor([y]) / (IMG_RES - 1))
                assert torch.isclose(model.pos_embd[0, y, x], expected_ch0, atol=1e-5)
                assert torch.isclose(model.pos_embd[1, y, x], expected_ch1, atol=1e-5)
                assert torch.isclose(model.pos_embd[2, y, x], expected_ch2, atol=1e-5)
                assert torch.isclose(model.pos_embd[3, y, x], expected_ch3, atol=1e-5)

    #TODO: When more than 4 channels are used for sinusoidal, the frequencies should be multiples of the base frequency (2). 
    # freq_bands = 2.0 ** np.linspace(0.0, num_freq, num=num_freq) is currently in code which gives
    # freqs = [1,4] instead of [1,2] for N_grid_channels=8. This seems to be a bug if we want the base 2.
    # Leaving it like this for now since we have checkpoints with 8 sinusoidal channels that use these frequencies,
    # but it should be fixed in the future and this test should be updated to reflect the intended behavior.
    def test_sinusoidal_8ch_grid_simple_values(self):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH + 8,
            out_channels=OUT_CH,
            gridtype="sinusoidal",
            N_grid_channels=8,
            **PE_SMALL_CFG,
        )
        # Check that the first 4 channels are sinusoids of different frequencies
        # and the next 4 channels are the cosine counterparts
        for y in range(IMG_RES):
            for x in range(IMG_RES):
                for idx, i in enumerate([0,2]):
                    expected_ch_0 = torch.sin((2**i) * 2 * torch.pi * torch.tensor([x]) / (IMG_RES - 1))
                    expected_ch_1 = torch.sin((2**i) * 2 * torch.pi * torch.tensor([y]) / (IMG_RES - 1))
                    expected_ch_2 = torch.cos((2**i) * 2 * torch.pi * torch.tensor([x]) / (IMG_RES - 1))
                    expected_ch_3 = torch.cos((2**i) * 2 * torch.pi * torch.tensor([y]) / (IMG_RES - 1))
                    assert torch.isclose(model.pos_embd[4*idx, y, x], expected_ch_0, atol=1e-5)
                    assert torch.isclose(model.pos_embd[4*idx + 1, y, x], expected_ch_1, atol=1e-5)
                    assert torch.isclose(model.pos_embd[4*idx + 2, y, x], expected_ch_2, atol=1e-5)
                    assert torch.isclose(model.pos_embd[4*idx + 3, y, x], expected_ch_3, atol=1e-5)


############################################################################
#                  SongUNetPosEmbd — forward                               #
############################################################################


class TestSongUNetPosEmbdForwardBasic:
    """Test basic forward pass for SongUNetPosEmbd."""

    def test_output_shape(self, small_pos_unet, noise_labels, class_labels):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        out = small_pos_unet(x, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_output_dtype_float32(self, small_pos_unet, noise_labels, class_labels):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        out = small_pos_unet(x, noise_labels, class_labels)
        assert out.dtype == torch.float32


class TestSongUNetPosEmbdForwardErrors:
    """Test that forward raises for mutually exclusive arguments."""

    def test_raises_when_both_selector_and_index_provided(
        self, small_pos_unet, noise_labels, class_labels
    ):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        global_index = torch.zeros(1, 2, IMG_RES, IMG_RES, dtype=torch.long)
        selector = lambda emb: emb[None].expand(B, -1, -1, -1)
        with pytest.raises(ValueError, match="Cannot provide both"):
            small_pos_unet(
                x, noise_labels, class_labels,
                global_index=global_index,
                embedding_selector=selector,
            )

    def test_raises_when_lead_time_mode_and_embedding_selector_provided(self, small_pos_unet, noise_labels, class_labels):
        small_pos_unet.lead_time_mode = True
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        selector = lambda emb: emb[None].expand(B, -1, -1, -1)
        with pytest.raises(ValueError, match="Embedding selector is not supported in lead time mode."):
            small_pos_unet(
                x, noise_labels, class_labels,
                embedding_selector=selector,
            )


class TestSongUNetPosEmbdForwardSelector:
    """Test forward with embedding_selector."""

    def test_selector_applied(self, small_pos_unet, noise_labels, class_labels):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        selector = lambda emb: emb[None].expand(B, -1, -1, -1)
        out = small_pos_unet(
            x, noise_labels, class_labels, embedding_selector=selector
        )
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_selector_takes_subset_of_embeddings(self, small_pos_unet, noise_labels, class_labels):
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        # Selector that takes only the first 2 channels of the positional embedding
        selector = lambda emb: emb[None].expand(B * P, -1, -1, -1)[:,:,:IMG_RES//2,:IMG_RES//2]
        noise_labels = torch.randn(B * P)
        class_labels = torch.randint(0, 1, (B * P, 1)).float()
        out = small_pos_unet(
            x, noise_labels, class_labels, embedding_selector=selector
        )
        assert out.shape == (B*P, OUT_CH, IMG_RES//2, IMG_RES//2)


class TestSongUNetPosEmbdForwardGlobalIndex:
    """Test forward with global_index."""

    def test_global_index_selects_embeddings(
        self, small_pos_unet, noise_labels, class_labels
    ):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        # Create index that selects the full grid
        idx_y = torch.arange(IMG_RES).view(1, 1, IMG_RES, 1).expand(1, 1, IMG_RES, IMG_RES)
        idx_x = torch.arange(IMG_RES).view(1, 1, 1, IMG_RES).expand(1, 1, IMG_RES, IMG_RES)
        global_index = torch.cat([idx_y, idx_x], dim=1)  # (P, 2, H, W)
        out = small_pos_unet(x, noise_labels, class_labels, global_index=global_index)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_global_index_selects_subset_of_embeddings(
        self, small_pos_unet
    ):
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        # Create index that selects only the top-left quadrant of the grid
        idx_y = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(P, 1, IMG_RES//2, IMG_RES//2)
        idx_x = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(P, 1, IMG_RES//2, IMG_RES//2)
        global_index = torch.cat([idx_y, idx_x], dim=1)  # (P, 2, H, W)
        noise_labels = torch.randn(B * P)
        class_labels = torch.randint(0, 1, (B * P, 1)).float()
        out = small_pos_unet(x, noise_labels, class_labels, global_index=global_index)
        assert out.shape == (B * P, OUT_CH, IMG_RES//2, IMG_RES//2)


class TestSongUNetPosEmbdForwardLeadTime:
    """Test forward pass with lead_time_mode enabled."""

    def _make_lead_time_model(self):
        lt_ch = 2
        return SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + lt_ch,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=lt_ch,
            lead_time_steps=5,
            prob_channels=[],
            **PE_SMALL_CFG,
        )

    def test_lead_time_forward_shape(self, noise_labels, class_labels):
        model = self._make_lead_time_model()
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        lead_time = torch.zeros(B, dtype=torch.long)
        out = model(x, noise_labels, class_labels, lead_time_label=lead_time)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)

    def test_lead_time_with_prob_channels_eval(self, noise_labels, class_labels):
        """In eval mode, prob_channels should go through softmax."""
        lt_ch = 2
        out_ch = 4
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + lt_ch,
            out_channels=out_ch,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=lt_ch,
            lead_time_steps=5,
            prob_channels=[2, 3],
            **PE_SMALL_CFG,
        )
        model.eval()
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        lead_time = torch.zeros(B, dtype=torch.long)
        out = model(x, noise_labels, class_labels, lead_time_label=lead_time)
        assert out.shape == (B, out_ch, IMG_RES, IMG_RES)
        # Prob channels should sum to 1 (softmax)
        prob_sum = out[:, [2, 3]].sum(dim=1)
        torch.testing.assert_close(
            prob_sum, torch.ones(B, IMG_RES, IMG_RES), atol=1e-5, rtol=1e-5
        )

    def test_lead_time_with_prob_channels_train(self, noise_labels, class_labels):
        """In training mode, prob_channels should output raw logits (no softmax)."""
        lt_ch = 2
        out_ch = 4
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + lt_ch,
            out_channels=out_ch,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=lt_ch,
            lead_time_steps=5,
            prob_channels=[2, 3],
            **PE_SMALL_CFG,
        )
        model.train()
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        lead_time = torch.zeros(B, dtype=torch.long)
        out = model(x, noise_labels, class_labels, lead_time_label=lead_time)
        assert out.shape == (B, out_ch, IMG_RES, IMG_RES)

    def test_lead_time_with_global_index(self, noise_labels, class_labels):
        """Test that global_index can be used with lead_time_mode."""
        lt_ch = 2
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + lt_ch,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            lead_time_mode=True,
            lead_time_channels=lt_ch,
            lead_time_steps=5,
            prob_channels=[],
            **PE_SMALL_CFG,
        )
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        # Create index that selects only the top-left quadrant of the grid
        idx_y = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(P, 1, IMG_RES//2, IMG_RES//2)
        idx_x = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(P, 1, IMG_RES//2, IMG_RES//2)
        global_index = torch.cat([idx_y, idx_x], dim=1)  # (P, 2, H, W)
        noise_labels = torch.randn(B * P)
        class_labels = torch.randint(0, 1, (B * P, 1)).float()
        out = model(x, noise_labels, class_labels, global_index=global_index, lead_time_label=torch.zeros(B, dtype=torch.long))
        assert out.shape == (B * P, OUT_CH, IMG_RES//2, IMG_RES//2)


class TestSongUNetPosEmbdForwardNoneGrid:
    """Test forward pass when N_grid_channels=0 (no positional embedding)."""

    def test_no_pos_embd_forward(self, noise_labels, class_labels):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=0,
            **PE_SMALL_CFG,
        )
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        out = model(x, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, IMG_RES, IMG_RES)


class TestSongUNetPosEmbdForwardRectangular:
    """Test forward with non-square resolution."""

    def test_rectangular_forward(self, noise_labels, class_labels):
        res = [16, 32]
        model = SongUNetPosEmbd(
            img_resolution=res,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            N_grid_channels=N_GRID,
            **PE_SMALL_CFG,
        )
        x = torch.randn(B, IN_CH, res[0], res[1])
        out = model(x, noise_labels, class_labels)
        assert out.shape == (B, OUT_CH, res[0], res[1])


############################################################################
#           SongUNetPosEmbd — positional_embedding_indexing                #
############################################################################


class TestPositionalEmbeddingIndexing:
    """Test positional_embedding_indexing method."""

    def test_no_index_returns_full_grid(self, small_pos_unet):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        result = small_pos_unet.positional_embedding_indexing(x)
        assert result.shape == (B, N_GRID, IMG_RES, IMG_RES)

    def test_no_index_expands_batch(self, small_pos_unet):
        x = torch.randn(4, IN_CH, IMG_RES, IMG_RES)
        result = small_pos_unet.positional_embedding_indexing(x)
        assert result.shape[0] == 4

    def test_global_index_selects_correctly(self, small_pos_unet):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            **PE_SMALL_CFG,
        )
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        idx_y = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(P, 1, IMG_RES//2, IMG_RES//2)
        idx_x = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(P, 1, IMG_RES//2, IMG_RES//2)
        global_index = torch.cat([idx_y, idx_x], dim=1)
        result = model.positional_embedding_indexing(x, global_index=global_index)
        assert result.shape == (B * P, 2, IMG_RES//2, IMG_RES//2)
        assert torch.allclose(result, model.pos_embd[None, :, :IMG_RES//2, :IMG_RES//2].expand(B*P, -1, -1, -1))

    def test_global_index_selects_correctly_with_lead_time(self, small_pos_unet):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            lead_time_mode=True,
            lead_time_channels=2,
            lead_time_steps=5,
            prob_channels=[],
            **PE_SMALL_CFG,
        )
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        idx_y = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(P, 1, IMG_RES//2, IMG_RES//2)
        idx_x = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(P, 1, IMG_RES//2, IMG_RES//2)
        global_index = torch.cat([idx_y, idx_x], dim=1)
        result = model.positional_embedding_indexing(x, global_index=global_index, lead_time_label=torch.zeros(B, dtype=torch.long))
        assert result.shape == (B * P, 2 + 2, IMG_RES//2, IMG_RES//2)
        expected_pos_embd = model.pos_embd[None, :, :IMG_RES//2, :IMG_RES//2].expand(B*P, -1, -1, -1)
        expected_lt_embd = model.lt_embd[0:1,:,:IMG_RES//2, :IMG_RES//2].expand(B*P, -1, -1, -1)  # Assuming lead_time_label=0 for this test
        expected_combined = torch.cat([expected_pos_embd, expected_lt_embd], dim=1)
        assert torch.allclose(result, expected_combined)

    def test_global_index_stacks_per_batch_elements(self, small_pos_unet):
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        idx_y_1 = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_x_1 = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_y_2 = torch.arange(IMG_RES//2, IMG_RES).view(1, 1, IMG_RES//2, 1).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_x_2 = torch.arange(IMG_RES//2, IMG_RES).view(1, 1, 1, IMG_RES//2).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_1 = torch.cat([idx_y_1, idx_x_1], dim=1)
        idx_2 = torch.cat([idx_y_2, idx_x_2], dim=1)
        global_index = torch.cat([idx_1, idx_2], dim=0)
        result = small_pos_unet.positional_embedding_indexing(x, global_index=global_index)
        assert result.shape == (B * P, N_GRID, IMG_RES//2, IMG_RES//2)
        # Check that the same positional embedding is repeated for each batch element in the group of P
        for i in range(B):
            assert torch.allclose(result[i*P], small_pos_unet.pos_embd[:, :IMG_RES//2, :IMG_RES//2])
            assert torch.allclose(result[i*P + 1], small_pos_unet.pos_embd[:, IMG_RES//2:, IMG_RES//2:])

    def test_global_index_stacks_per_batch_elements_with_lead_time(self, small_pos_unet):
        model = SongUNetPosEmbd(
            img_resolution=IMG_RES,
            in_channels=PE_IN_CH + 2,
            out_channels=OUT_CH,
            gridtype="linear",
            N_grid_channels=2,
            lead_time_mode=True,
            lead_time_channels=2,
            lead_time_steps=5,
            prob_channels=[],
            **PE_SMALL_CFG,
        )
        P = 2
        x = torch.randn(B * P, IN_CH, IMG_RES//2, IMG_RES//2)
        idx_y_1 = torch.arange(IMG_RES//2).view(1, 1, IMG_RES//2, 1).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_x_1 = torch.arange(IMG_RES//2).view(1, 1, 1, IMG_RES//2).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_y_2 = torch.arange(IMG_RES//2, IMG_RES).view(1, 1, IMG_RES//2, 1).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_x_2 = torch.arange(IMG_RES//2, IMG_RES).view(1, 1, 1, IMG_RES//2).expand(1, 1, IMG_RES//2, IMG_RES//2)
        idx_1 = torch.cat([idx_y_1, idx_x_1], dim=1)
        idx_2 = torch.cat([idx_y_2, idx_x_2], dim=1)
        global_index = torch.cat([idx_1, idx_2], dim=0)
        result = model.positional_embedding_indexing(x, global_index=global_index, lead_time_label=torch.zeros(B, dtype=torch.long))
        assert result.shape == (B * P, 4, IMG_RES//2, IMG_RES//2)  # Assuming pos_embd has 2 channels and lt_embd has 2 channels
        expected_pos_embd = model.pos_embd[None,::]
        expected_lt_embd = model.lt_embd[0:1] # Assuming lead_time_label=0 for this test
        expected_combined = torch.cat([expected_pos_embd, expected_lt_embd], dim=1)
        for i in range(B):
            assert torch.allclose(result[i*P], expected_combined[0, :, :IMG_RES//2, :IMG_RES//2])
            assert torch.allclose(result[i*P + 1], expected_combined[0, :, IMG_RES//2:, IMG_RES//2:])

    def test_dtype_conversion(self, small_pos_unet):
        """Embedding dtype should match input dtype."""
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES, dtype=torch.float64)
        result = small_pos_unet.positional_embedding_indexing(x)
        assert result.dtype == torch.float64


############################################################################
#           SongUNetPosEmbd — positional_embedding_selector                #
############################################################################


class TestPositionalEmbeddingSelector:
    """Test positional_embedding_selector method."""

    def test_selector_identity(self, small_pos_unet):
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES)
        selector = lambda emb: emb[None].expand(B, -1, -1, -1)
        result = small_pos_unet.positional_embedding_selector(x, selector)
        assert result.shape == (B, N_GRID, IMG_RES, IMG_RES)

    def test_selector_dtype_conversion(self, small_pos_unet):
        """Embedding dtype should be cast to input dtype before selector runs."""
        x = torch.randn(B, IN_CH, IMG_RES, IMG_RES, dtype=torch.float64)
        selector = lambda emb: emb[None].expand(B, -1, -1, -1)
        result = small_pos_unet.positional_embedding_selector(x, selector)
        assert result.dtype == torch.float64

    def test_selector_returns_custom_shape(self, small_pos_unet):
        """Selector can return patches of a different spatial size."""
        patch_h, patch_w = 8, 8
        selector = lambda emb: emb[None, :, :patch_h, :patch_w].expand(B, -1, -1, -1)
        x = torch.randn(B, IN_CH, patch_h, patch_w)
        result = small_pos_unet.positional_embedding_selector(x, selector)
        assert result.shape == (B, N_GRID, patch_h, patch_w)
        assert torch.allclose(result, small_pos_unet.pos_embd[None, :, :patch_h, :patch_w].expand(B, -1, -1, -1))
