from __future__ import annotations

from pathlib import Path

import numpy as np

from eval._backends.weight_diagnostics import plot_checkpoint_weights as mod


def test_build_arg_parser_accepts_all_supported_lanes():
    parser = mod.build_arg_parser()

    for lane in ("o48_o96", "o96_o320", "o320_o1280", "o1280_o2560"):
        args = parser.parse_args(
            ["--ckpt-path", "/tmp/demo.ckpt", "--out-dir", "/tmp/out", "--lane", lane]
        )
        assert args.lane == lane


def test_lane_metadata_uses_target_only_defaults_for_newer_lanes():
    assert mod.lane_references("o48_o96") == {}
    assert mod.lane_thresholds("o48_o96") is None
    assert mod.lane_references("o1280_o2560") == {}
    assert mod.lane_thresholds("o1280_o2560") is None


def test_plot_weight_distributions_supports_target_only_mode(tmp_path: Path):
    out_path = mod.plot_weight_distributions(
        [
            {
                "label": "target",
                "role": "target",
                "decoder_attn": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "decoder_mlp": np.array([0.2, 0.4, 0.6], dtype=np.float32),
                "proc_attn": np.array([-0.1, 0.0, 0.1], dtype=np.float32),
                "proc_mlp": np.array([0.05, 0.15, 0.25], dtype=np.float32),
            }
        ],
        tmp_path,
    )

    assert out_path == tmp_path / "weight_distributions.png"
    assert out_path.exists()


def test_plot_norm_summary_supports_lane_without_thresholds(tmp_path: Path):
    out_path = mod.plot_norm_summary(
        [
            {
                "label": "target",
                "role": "target",
                "family_norms": {
                    "output_head": 4.0,
                    "decoder.attn": 12.5,
                    "encoder.attn": 3.0,
                    "processor.attn": 2.5,
                    "decoder.mlp": 7.0,
                    "encoder.mlp": 6.5,
                    "processor.mlp": 5.0,
                    "decoder.embed": 1.5,
                    "encoder.embed": 1.0,
                    "noise_cond": 0.8,
                },
            }
        ],
        "o48_o96",
        tmp_path,
    )

    assert out_path == tmp_path / "weight_norms_by_family.png"
    assert out_path.exists()
