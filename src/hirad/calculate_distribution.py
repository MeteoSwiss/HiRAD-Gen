"""
Calculate wind (10u, 10v) distributions from the AnemoiDataset.

Iterates through the full dataset, bins each gridpoint value into
0.5 m/s wide bins from -100 to 100 m/s, and saves:
  - histogram counts as .npy files
  - a combined 2x2 bar-chart figure as .png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

from hirad.datasets.anemoi_dataset import AnemoiDataset

# ── Hardcoded dataset parameters (from conf/dataset/anemoi_era_real.yaml) ────
DATASET_PARAMS = dict(
    type="anemoi_era5_real",
    input_anemoi_dataset_path="/capstor/store/mch/msopr/ml/datasets/"
        "aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51.zarr",
    target_anemoi_dataset_path="/capstor/store/mch/msopr/ml/datasets/"
        "mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr",
    input_channel_names=[
        "2t", "10u", "10v", "tcw",
        "t_850", "z_850", "u_850", "v_850",
        "t_500", "z_500", "u_500", "v_500",
        "tp",
    ],
    output_channel_names=["2t", "10u", "10v", "tp"],
    static_channel_names=["FIS"],
    transform_channels=["tp-box_cox_025"],
    transform_input_means={"tp-box_cox_025": -3.815209745618941},
    transform_input_stdevs={"tp-box_cox_025": 0.22851179478814418},
    transform_output_means={"tp-box_cox_025": -3.8121187083242556},
    transform_output_stdevs={"tp-box_cox_025": 0.3345851858215482},
    n_month_hour_channels=4,
    start_date="2023-01-01",
    end_date="2023-12-31",
    trim_edge=41,
)

# ── Wind channel indices ─────────────────────────────────────────────────────
# input_channel_names:  [2t, 10u, 10v, ...]  → 10u=1, 10v=2
# output_channel_names: [2t, 10u, 10v, tp]   → 10u=1, 10v=2
INPUT_U_IDX = 1
INPUT_V_IDX = 2
OUTPUT_U_IDX = 1
OUTPUT_V_IDX = 2

# ── Histogram parameters ─────────────────────────────────────────────────────
BIN_EDGES = np.arange(-100, 100.5, 0.5)   # 401 edges → 400 bins of width 0.5
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "distributions"


def _process_chunk(worker_id, indices, dataset):
    """Worker: build partial histograms + running sums for a chunk of indices."""


    iu_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    iv_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    ou_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    ov_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)

    # Running sums for mean / variance (Welford-style two-pass is overkill
    # for histogramming; we use simple sum / sum-of-squares instead).
    iu_sum = 0.0;  iu_sq = 0.0;  iu_n = 0
    iv_sum = 0.0;  iv_sq = 0.0;  iv_n = 0
    ou_sum = 0.0;  ou_sq = 0.0;  ou_n = 0
    ov_sum = 0.0;  ov_sq = 0.0;  ov_n = 0

    iterator = tqdm(indices, desc="Worker 0", unit="sample") if worker_id == 0 else indices
    for i in iterator:
        target, input_data, _ = dataset[i]

        iu = input_data[INPUT_U_IDX].numpy().ravel()
        iv = input_data[INPUT_V_IDX].numpy().ravel()
        ou = target[OUTPUT_U_IDX].numpy().ravel()
        ov = target[OUTPUT_V_IDX].numpy().ravel()

        iu_hist += np.histogram(iu, bins=BIN_EDGES)[0]
        iv_hist += np.histogram(iv, bins=BIN_EDGES)[0]
        ou_hist += np.histogram(ou, bins=BIN_EDGES)[0]
        ov_hist += np.histogram(ov, bins=BIN_EDGES)[0]

        iu_sum += iu.sum(); iu_sq += (iu ** 2).sum(); iu_n += iu.size
        iv_sum += iv.sum(); iv_sq += (iv ** 2).sum(); iv_n += iv.size
        ou_sum += ou.sum(); ou_sq += (ou ** 2).sum(); ou_n += ou.size
        ov_sum += ov.sum(); ov_sq += (ov ** 2).sum(); ov_n += ov.size

    return (
        iu_hist, iv_hist, ou_hist, ov_hist,
        iu_sum, iu_sq, iu_n,
        iv_sum, iv_sq, iv_n,
        ou_sum, ou_sq, ou_n,
        ov_sum, ov_sq, ov_n,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Create the dataset ────────────────────────────────────────────────
    print("Initialising AnemoiDataset …")
    dataset = AnemoiDataset(**DATASET_PARAMS)
    n_samples = len(dataset)
    print(f"Dataset ready – {n_samples} time steps.")

    # ── 2. Split work across threads ──────────────────────────────────────────
    n_workers = min(os.cpu_count(), 16)
    chunks = np.array_split(np.arange(n_samples), n_workers)
    chunks = [c.tolist() for c in chunks if len(c) > 0]
    print(f"Computing histograms with {len(chunks)} threads …")

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_process_chunk, wid, chunk, dataset)
            for wid, chunk in enumerate(chunks)
        ]
        results = [f.result() for f in futures]

    # ── 3. Reduce partial results ────────────────────────────────────────────
    input_u_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    input_v_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    output_u_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)
    output_v_hist = np.zeros(len(BIN_CENTERS), dtype=np.int64)

    iu_sum = iv_sum = ou_sum = ov_sum = 0.0
    iu_sq = iv_sq = ou_sq = ov_sq = 0.0
    iu_n = iv_n = ou_n = ov_n = 0

    for r in results:
        (r_iu_h, r_iv_h, r_ou_h, r_ov_h,
         r_iu_s, r_iu_sq, r_iu_n,
         r_iv_s, r_iv_sq, r_iv_n,
         r_ou_s, r_ou_sq, r_ou_n,
         r_ov_s, r_ov_sq, r_ov_n) = r

        input_u_hist += r_iu_h;  iu_sum += r_iu_s;  iu_sq += r_iu_sq;  iu_n += r_iu_n
        input_v_hist += r_iv_h;  iv_sum += r_iv_s;  iv_sq += r_iv_sq;  iv_n += r_iv_n
        output_u_hist += r_ou_h; ou_sum += r_ou_s;  ou_sq += r_ou_sq;  ou_n += r_ou_n
        output_v_hist += r_ov_h; ov_sum += r_ov_s;  ov_sq += r_ov_sq;  ov_n += r_ov_n

    # Compute global mean & std
    def _mean_std(s, sq, n):
        mean = s / n
        std = np.sqrt(sq / n - mean ** 2)
        return mean, std

    iu_mean, iu_std = _mean_std(iu_sum, iu_sq, iu_n)
    iv_mean, iv_std = _mean_std(iv_sum, iv_sq, iv_n)
    ou_mean, ou_std = _mean_std(ou_sum, ou_sq, ou_n)
    ov_mean, ov_std = _mean_std(ov_sum, ov_sq, ov_n)

    print(f"Input  10u  – mean={iu_mean:.4f}, std={iu_std:.4f}")
    print(f"Input  10v  – mean={iv_mean:.4f}, std={iv_std:.4f}")
    print(f"Output 10u  – mean={ou_mean:.4f}, std={ou_std:.4f}")
    print(f"Output 10v  – mean={ov_mean:.4f}, std={ov_std:.4f}")

    # ── 4. Save histogram data ───────────────────────────────────────────────
    np.save(OUTPUT_DIR / "input_10u_hist.npy", input_u_hist)
    np.save(OUTPUT_DIR / "input_10v_hist.npy", input_v_hist)
    np.save(OUTPUT_DIR / "output_10u_hist.npy", output_u_hist)
    np.save(OUTPUT_DIR / "output_10v_hist.npy", output_v_hist)
    np.save(OUTPUT_DIR / "bin_edges.npy", BIN_EDGES)
    np.save(OUTPUT_DIR / "bin_centers.npy", BIN_CENTERS)
    print(f"Histogram data saved to {OUTPUT_DIR}")

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    histograms = [
        ("Input 10u",  input_u_hist,  iu_mean, iu_std),
        ("Input 10v",  input_v_hist,  iv_mean, iv_std),
        ("Output 10u", output_u_hist, ou_mean, ou_std),
        ("Output 10v", output_v_hist, ov_mean, ov_std),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, counts, mean, std) in zip(axes.flat, histograms):
        ax.bar(BIN_CENTERS, counts, width=0.5, edgecolor="black", linewidth=0.2)
        ax.set_title(title)
        ax.set_xlabel("Wind speed (m/s)")
        ax.set_ylabel("Count")
        ax.set_xlim(-50, 50)
        ax.text(
            0.97, 0.95,
            f"mean = {mean:.4f}\nstd  = {std:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.suptitle(f"Wind (10u / 10v) value distributions\n{DATASET_PARAMS['start_date']} to {DATASET_PARAMS['end_date']}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wind_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {OUTPUT_DIR / 'wind_distributions.png'}")


if __name__ == "__main__":
    main()
