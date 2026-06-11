#!/usr/bin/env python3
"""
Weight distribution diagnostics for checkpoint evaluation.

CLI entry point:
    python -m eval._backends.weight_diagnostics.plot_checkpoint_weights

IMPORTANT: Must be run with .ds-dyn virtual environment (not .ds-multi).
Training checkpoints cannot be loaded with the newer anemoi-models in .ds-multi.

Outputs (written to --out-dir):
    weight_distributions.png   — 4x2 KDE grid
    weight_norms_by_family.png — horizontal grouped bar chart with Goldilocks zone bands
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lane metadata
# ---------------------------------------------------------------------------
SUPPORTED_LANES = (
    "o48_o96",
    "o96_o320",
    "o320_o1280",
    "o1280_o2560",
)

LANE_METADATA: dict[str, dict[str, object]] = {
    "o48_o96": {
        "references": {},
        "thresholds": None,
    },
    "o96_o320": {
        "references": {
            "high_wd": {
                "label": "39991df8 100k wd=0.1 [high-wd ref]",
                "path": "/home/ecm5702/scratch/aifs/checkpoint/39991df81216460fb7f3bd048df733c3/anemoi-by_epoch-epoch_021-step_100000.ckpt",
            },
            "low_wd": {
                "label": "cfec83a3 200k wd=0.01 [low-wd/best ref]",
                "path": "/home/ecm5702/scratch/aifs/checkpoint/cfec83a3cd0644778e2bfcbacfa9f4fc/anemoi-by_epoch-epoch_043-step_200000.ckpt",
            },
        },
        "thresholds": {
            "boundaries": [40.0, 55.0, 70.0],
            "zones": ["suppressed", "transition", "Goldilocks", "overshoot"],
            "title_suffix": "",
        },
    },
    "o320_o1280": {
        "references": {
            "high_wd": {
                "label": "f03c9e88 100k wd=0.1 [high-wd ref]",
                "path": "/home/ecm5702/scratch/aifs/checkpoint/f03c9e8862c9422f972629814d94a3a3/anemoi-by_step-epoch_021-step_100000.ckpt",
            },
            "low_wd": {
                "label": "da4d902b 100k wd=0.01 [low-wd ref]",
                "path": "/home/ecm5702/scratch/aifs/checkpoint/da4d902b71084ecc884a938c4b8930d3/anemoi-by_step-epoch_021-step_100000.ckpt",
            },
        },
        "thresholds": {
            "boundaries": [40.0, 55.0, 70.0],
            "zones": ["suppressed", "transition", "Goldilocks", "overshoot"],
            "title_suffix": " (provisional, o320->o1280)",
        },
    },
    "o1280_o2560": {
        "references": {},
        "thresholds": None,
    },
}

# ---------------------------------------------------------------------------
# Weight family classification — copied verbatim from extract_weight_norms.py
# ---------------------------------------------------------------------------
BUFFER_PATTERNS = re.compile(
    r"edge_inc|latlons|_norm_mul|_norm_add|_input_lres_idx|_output_idx|\.frequencies$"
    r"|normalizer_input\._|normalizer_output\._"
)

# Module family rules (first match wins). All keys live under model.*
FAMILY_RULES = [
    # Output projection head
    ("output_head",    r"model\.model\.decoder\.node_data_extractor"),
    # Attention projections in each component
    ("decoder.attn",   r"model\.model\.decoder\..*\.(lin_query|lin_key|lin_value|projection)"),
    ("encoder.attn",   r"model\.model\.encoder\..*\.(lin_query|lin_key|lin_value|projection)"),
    ("processor.attn", r"model\.model\.processor\..*\.(lin_query|lin_key|lin_value|projection)"),
    # MLP blocks inside each component
    ("decoder.mlp",    r"model\.model\.decoder\.proc\.(node_dst_mlp|node_src_mlp|edges_mlp|mlp)\b"),
    ("encoder.mlp",    r"model\.model\.encoder\.proc\.(node_dst_mlp|node_src_mlp|edges_mlp|mlp)\b"),
    ("processor.mlp",  r"model\.model\.processor\.proc\.[^.]*\.(node_dst_mlp|node_src_mlp|edges_mlp|mlp)\b"),
    # Node embeddings
    ("decoder.embed",  r"model\.model\.decoder\.emb_nodes"),
    ("encoder.embed",  r"model\.model\.encoder\.emb_nodes"),
    # Noise conditioning network
    ("noise_cond",     r"model\.model\.noise_cond_mlp|model\.model\.noise_embedder"),
    # Full-component catch-alls (for remaining proc sub-layers not caught above)
    ("decoder",        r"model\.model\.decoder"),
    ("encoder",        r"model\.model\.encoder"),
    ("processor",      r"model\.model\.processor"),
    # Preprocessing/postprocessing (not weight-decay sensitive — informational only)
    ("pre_post_proc",  r"model\.(pre|post)_processors"),
]

FAMILY_PATTERNS = [(name, re.compile(pat)) for name, pat in FAMILY_RULES]

# Families shown in the norm summary bar chart (in display order)
NORM_SUMMARY_FAMILIES = [
    "output_head",
    "decoder.attn",
    "encoder.attn",
    "processor.attn",
    "decoder.mlp",
    "encoder.mlp",
    "processor.mlp",
    "decoder.embed",
    "encoder.embed",
    "noise_cond",
]

# KDE panel extraction patterns
_DECODER_ATTN_PATTERNS   = ["lin_key.weight", "lin_query.weight", "lin_value.weight",
                              "lin_self.weight", "projection.weight"]
_DECODER_MLP_PATTERNS    = ["node_dst_mlp", "node_src_mlp", "edges_mlp"]
_PROCESSOR_ATTN_PATTERNS = ["lin_key.weight", "lin_query.weight", "lin_value.weight",
                              "projection.weight"]
_PROCESSOR_MLP_PATTERNS  = ["node_dst_mlp", "node_src_mlp", "edges_mlp"]

_RNG = np.random.default_rng(42)
_KDE_SUBSAMPLE = 80_000


# ---------------------------------------------------------------------------
# Lane metadata helpers
# ---------------------------------------------------------------------------

def lane_metadata(lane: str) -> dict[str, object]:
    """Return the configured metadata for *lane*."""
    return LANE_METADATA[lane]


def lane_references(lane: str) -> dict[str, dict[str, str]]:
    """Return trusted reference checkpoint definitions for *lane*."""
    return lane_metadata(lane)["references"]  # type: ignore[return-value]


def lane_thresholds(lane: str) -> dict[str, object] | None:
    """Return threshold-band metadata for *lane*, if documented."""
    return lane_metadata(lane)["thresholds"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Classification helper
# ---------------------------------------------------------------------------

def _classify(key: str) -> tuple[str, bool]:
    """Return (family, is_buffer). Buffers are excluded from norm analysis."""
    if BUFFER_PATTERNS.search(key):
        return "excluded_buffer", True
    for name, pat in FAMILY_PATTERNS:
        if pat.search(key):
            return name, False
    return "other", False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_weights(state_dict: dict, patterns: list[str], scope: str) -> np.ndarray:
    """Flatten all matching weight tensors in *scope* into a 1-D float32 array."""
    vecs = []
    for key, tensor in state_dict.items():
        if scope not in key:
            continue
        if not key.endswith(".weight"):
            continue
        if any(p in key for p in patterns):
            vecs.append(tensor.float().cpu().numpy().ravel())
    if not vecs:
        return np.array([], dtype=np.float32)
    return np.concatenate(vecs)


def compute_family_norms(state_dict: dict) -> dict[str, float | None]:
    """Return mean per-tensor L2 norm for each family in NORM_SUMMARY_FAMILIES."""
    family_norms: dict[str, list[float]] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        fam, is_buffer = _classify(key)
        if is_buffer or fam not in NORM_SUMMARY_FAMILIES:
            continue
        family_norms.setdefault(fam, []).append(tensor.float().norm().item())
    result = {}
    for fam in NORM_SUMMARY_FAMILIES:
        ns = family_norms.get(fam, [])
        result[fam] = sum(ns) / len(ns) if ns else None
    return result


def plot_weight_distributions(data: list[dict], out_dir: Path | str) -> Path:
    """
    4x2 KDE grid. Rows = (decoder_attn, decoder_mlp, proc_attn, proc_mlp).
    Left col = full wd=0.01 scale (99.9th pct of low-wd ref).
    Right col = zoomed wd=0.1 scale (99.9th pct of high-wd ref).

    data entries must have keys:
    label, role, decoder_attn, decoder_mlp, proc_attn, proc_mlp.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _style: dict[str, tuple[str, str]] = {
        "target":      ("black",   "-"),
        "high_wd_ref": ("red",     "--"),
        "low_wd_ref":  ("#1565C0", "-."),
    }

    panel_rows = [
        ("decoder_attn", "Decoder attention"),
        ("decoder_mlp",  "Decoder MLP"),
        ("proc_attn",    "Processor attention"),
        ("proc_mlp",     "Processor MLP"),
    ]

    def _concat_role(key: str, role: str) -> np.ndarray:
        arrays = [d[key] for d in data if d.get("role") == role and d[key].size > 0]
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float32)

    def _pct_range(arr: np.ndarray, pct: float = 99.9) -> tuple[float, float]:
        return float(np.percentile(arr, 100 - pct)), float(np.percentile(arr, pct))

    have_reference_overlays = any(d.get("role") != "target" for d in data)
    fig, axes = plt.subplots(len(panel_rows), 2, figsize=(16, 16))
    if have_reference_overlays:
        fig.suptitle(
            "Scalar weight distributions — checkpoint vs references\n"
            "Left: full wd=0.01 scale  |  Right: zoomed into wd=0.1 range",
            fontsize=11,
        )
        left_title_suffix = "full scale (wd=0.01 range)"
        right_title_suffix = "zoom (wd=0.1 range)"
    else:
        fig.suptitle(
            "Scalar weight distributions — target checkpoint only\n"
            "Left: broad percentile range  |  Right: central percentile zoom",
            fontsize=11,
        )
        left_title_suffix = "broad range"
        right_title_suffix = "central zoom"

    for row_idx, (key, family_label) in enumerate(panel_rows):
        low_wd_arr  = _concat_role(key, "low_wd_ref")
        high_wd_arr = _concat_role(key, "high_wd_ref")
        all_arrs = [d[key] for d in data if d[key].size > 0]
        if not all_arrs:
            continue
        all_arr = np.concatenate(all_arrs)

        wide_range = _pct_range(low_wd_arr  if low_wd_arr.size  > 0 else all_arr)
        zoom_range = _pct_range(high_wd_arr if high_wd_arr.size > 0 else all_arr)

        ax_w, ax_z = axes[row_idx, 0], axes[row_idx, 1]

        for d in data:
            arr = d[key]
            if arr.size == 0:
                continue
            color, ls = _style.get(d.get("role", "target"), ("grey", "-"))
            _plot_kde(ax_w, arr, label=d["label"], color=color, ls=ls,
                      x_lo=wide_range[0], x_hi=wide_range[1])
            _plot_kde(ax_z, arr, label=d["label"], color=color, ls=ls,
                      x_lo=zoom_range[0], x_hi=zoom_range[1])

        for ax, title in [
            (ax_w, f"{family_label} — {left_title_suffix}"),
            (ax_z, f"{family_label} — {right_title_suffix}"),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Weight value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
            ax.tick_params(labelsize=8)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, framealpha=0.85)

    plt.tight_layout()
    out_path = out_dir / "weight_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)
    return out_path


def plot_norm_summary(data: list[dict], lane: str, out_dir: Path | str) -> Path:
    """
    Horizontal grouped bar chart: one group per family, bars per checkpoint.
    Overlays Goldilocks threshold zones across the full chart.
    Title identifies which zone the target decoder.attn norm falls in.

    data entries must have keys: label, role, family_norms.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = lane_thresholds(lane)
    boundaries: list[float] = []
    zones: list[str] = []
    title_suffix = ""
    if thresholds is not None:
        boundaries = list(thresholds["boundaries"])  # type: ignore[index]
        zones = list(thresholds["zones"])  # type: ignore[index]
        title_suffix = str(thresholds["title_suffix"])  # type: ignore[index]

    role_style: dict[str, tuple[str, float]] = {
        "target":      ("black",   0.90),
        "high_wd_ref": ("red",     0.65),
        "low_wd_ref":  ("#1565C0", 0.65),
    }

    n_fam    = len(NORM_SUMMARY_FAMILIES)
    n_ds     = len(data)
    bar_h    = 0.25
    gap      = 0.20
    group_sz = n_ds * bar_h + gap

    fig, ax = plt.subplots(figsize=(14, max(6, n_fam * 1.2)))

    ytick_pos: list[float]  = []
    ytick_labels: list[str] = []
    max_val = 1.0

    for fi, fam in enumerate(NORM_SUMMARY_FAMILIES):
        gc_y = fi * group_sz
        for di, d in enumerate(data):
            v = d["family_norms"].get(fam)
            if v is None:
                continue
            max_val = max(max_val, v)
            y = gc_y + (di - (n_ds - 1) / 2.0) * bar_h
            color, alpha = role_style.get(d.get("role", "target"), ("grey", 0.7))
            ax.barh(
                y, v, height=bar_h * 0.85,
                color=color, alpha=alpha,
                label=d["label"] if fi == 0 else "_nolegend_",
            )
        ytick_pos.append(gc_y)
        ytick_labels.append(fam.replace(".", "_"))

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-tensor L2 norm", fontsize=10)

    x_max = max_val * 1.15
    if boundaries:
        x_max = max(x_max, boundaries[-1] * 1.3)
    ax.set_xlim(0, x_max)

    if boundaries:
        # Zone background bands (subtle shading)
        zone_bg = ["#ef9a9a", "#fff176", "#a5d6a7", "#ce93d8"]
        edges = [0.0] + boundaries + [x_max]
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            ax.axvspan(lo, hi, alpha=0.07, color=zone_bg[i % len(zone_bg)], zorder=0)

        # Threshold boundary lines
        for bnd in boundaries:
            ax.axvline(bnd, color="dimgrey", linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)

        # Zone name labels just above the top of the plot
        xform = ax.get_xaxis_transform()
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mid = (lo + hi) / 2.0
            if mid < x_max:
                ax.text(mid, 1.02, zones[i], transform=xform,
                        ha="center", va="bottom", fontsize=7, color="dimgrey")

        # Threshold values at boundary lines, slightly higher
        for bnd in boundaries:
            ax.text(bnd, 1.07, str(int(bnd)), transform=xform,
                    ha="center", va="bottom", fontsize=7, color="dimgrey")

    # Title: identify target's decoder.attn zone
    target = next((d for d in data if d.get("role") == "target"), None)
    zone_str = "decoder_attn norm: unavailable"
    if target:
        v = target["family_norms"].get("decoder.attn")
        if v is not None:
            if boundaries:
                z = zones[-1]
                for i, bnd in enumerate(boundaries):
                    if v < bnd:
                        z = zones[i]
                        break
                zone_str = f"decoder_attn = {v:.1f} -> {z}{title_suffix}"
            else:
                zone_str = f"decoder_attn = {v:.1f}"

    ax.set_title(f"Weight L2 norms by family  |  {zone_str}", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85, loc="lower right")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=2)

    plt.tight_layout()
    out_path = out_dir / "weight_norms_by_family.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_kde(
    ax: plt.Axes,
    values: np.ndarray,
    label: str,
    color: str,
    ls: str,
    x_lo: float,
    x_hi: float,
    n_pts: int = 800,
) -> None:
    clipped = values[(values >= x_lo) & (values <= x_hi)]
    if clipped.size < 2:
        return
    if clipped.size > _KDE_SUBSAMPLE:
        clipped = _RNG.choice(clipped, size=_KDE_SUBSAMPLE, replace=False)
    kde = gaussian_kde(clipped, bw_method=0.05)
    xs = np.linspace(x_lo, x_hi, n_pts)
    ax.plot(xs, kde(xs), color=color, linestyle=ls, linewidth=1.8, label=label, alpha=0.90)


def _load_ckpt_data(path: str, label: str) -> dict:
    log.info("Loading %s ...", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
    for k in list(ckpt.keys()):
        if k != "state_dict":
            del ckpt[k]
    del ckpt
    gc.collect()

    d = {
        "label":        label,
        "decoder_attn": extract_weights(sd, _DECODER_ATTN_PATTERNS,   "model.model.decoder"),
        "decoder_mlp":  extract_weights(sd, _DECODER_MLP_PATTERNS,    "model.model.decoder"),
        "proc_attn":    extract_weights(sd, _PROCESSOR_ATTN_PATTERNS, "model.model.processor"),
        "proc_mlp":     extract_weights(sd, _PROCESSOR_MLP_PATTERNS,  "model.model.processor"),
        "family_norms": compute_family_norms(sd),
    }
    del sd
    gc.collect()
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for checkpoint weight diagnostics."""
    ap = argparse.ArgumentParser(
        description=(
            "Weight distribution diagnostics for checkpoint evaluation.\n\n"
            "IMPORTANT: Must be run with .ds-dyn venv (not .ds-multi).\n"
            "Training checkpoints cannot be loaded with newer anemoi-models."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt-path",  required=True,
                    help="Absolute path to target checkpoint (.ckpt)")
    ap.add_argument("--ckpt-label", default=None,
                    help="Display label for the target checkpoint")
    ap.add_argument("--out-dir",    required=True,
                    help="Output directory for PNG files")
    ap.add_argument(
        "--lane",
        choices=SUPPORTED_LANES,
        default="o96_o320",
        help="Lane selects any documented references and threshold bands",
    )
    ap.add_argument(
        "--no-refs", action="store_true",
        help="Skip reference checkpoints; plot target only",
    )
    return ap


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.ckpt_label or Path(args.ckpt_path).name

    # Load target
    log.info("=== Target checkpoint ===")
    try:
        target = _load_ckpt_data(args.ckpt_path, label)
        target["role"] = "target"
    except Exception as exc:
        log.error("Failed to load target checkpoint: %s", exc)
        sys.exit(1)

    data: list[dict] = [target]

    # Load references unless suppressed
    if not args.no_refs:
        refs = lane_references(args.lane)
        if not refs:
            log.info("Lane %s has no documented default references; plotting target only.", args.lane)
        for role_key, ref_key in [("high_wd_ref", "high_wd"), ("low_wd_ref", "low_wd")]:
            ref = refs.get(ref_key)
            if ref is None:
                continue
            if not Path(ref["path"]).exists():
                log.warning(
                    "Reference checkpoint not found (scratch may be cleaned): %s — continuing without it",
                    ref["path"],
                )
                continue
            log.info("=== %s reference: %s ===", role_key, ref["label"])
            try:
                ref_data = _load_ckpt_data(ref["path"], ref["label"])
                ref_data["role"] = role_key
                data.append(ref_data)
            except Exception as exc:
                log.warning("Could not load %s: %s — skipping", role_key, exc)

    log.info("Loaded %d checkpoint(s). Generating plots in %s ...", len(data), out_dir)
    plot_weight_distributions(data, out_dir)
    plot_norm_summary(data, args.lane, out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
