"""Compute Box-Cox(0.25) normalization stats for the ifso1280_real `tp` channel.

Builds the exact training-split dataset (type=anemoi_ifso1280_real, same
start_date/end_date that training will use), applies the Box-Cox transform to the
raw physical-space input and output tp values, and reports mean/std to paste into
the dataset config's transform_input_means/transform_input_stdevs/
transform_output_means/transform_output_stdevs (key 'tp-box_cox_025').

Iterates dataset[idx] directly (no DataLoader/DistributedManager needed) over a
strided subsample of the training split to bound runtime - see --stride.
"""
import argparse
import time

import numpy as np

from hirad.datasets.dataset import known_datasets

INPUT_ANEMOI_DATASET_PATH = '/capstor/scratch/cscs/pstamenk/ifs-hres-realch1/ifs_hres_traj_2020_202502.zarr'
TARGET_ANEMOI_DATASET_PATH = '/capstor/store/mch/msopr/ml/datasets/mch-realch1-fdb-1km-2005-2025-1h-pl13-v1.0.zarr'
INPUT_CHANNEL_NAMES = ['2t', '10u', '10v', 'tcw', 't_850', 'z_850', 'u_850', 'v_850', 't_500', 'z_500', 'u_500', 'v_500', 'tp']
OUTPUT_CHANNEL_NAMES = ['2t', '10u', '10v', 'tp']
TP_INPUT_IDX = INPUT_CHANNEL_NAMES.index('tp')
TP_OUTPUT_IDX = OUTPUT_CHANNEL_NAMES.index('tp')
LAMBDA = 0.25


def box_cox(x: np.ndarray, lmbda: float = LAMBDA) -> np.ndarray:
    return (np.power(np.clip(x, 0, None), lmbda) - 1) / lmbda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date', default='2020-10-01')
    parser.add_argument('--end-date', default='2024-08-31')
    parser.add_argument('--stride', type=int, default=50, help='use every Nth (base_date, step) pair')
    parser.add_argument('--lead-time-hours', type=int, nargs='+', default=None,
                         help='restrict to these lead times (hours), e.g. --lead-time-hours 1')
    args = parser.parse_args()

    dataset_cfg = dict(
        type='anemoi_ifso1280_real',
        input_anemoi_dataset_path=INPUT_ANEMOI_DATASET_PATH,
        target_anemoi_dataset_path=TARGET_ANEMOI_DATASET_PATH,
        input_channel_names=INPUT_CHANNEL_NAMES,
        output_channel_names=OUTPUT_CHANNEL_NAMES,
        start_date=args.start_date,
        end_date=args.end_date,
        lead_time_hours=args.lead_time_hours,
    )
    dataset = known_datasets[dataset_cfg['type']](**dataset_cfg)

    indices = range(0, len(dataset), args.stride)
    print(f"Training split has {len(dataset)} (base_date, step) pairs; "
          f"using every {args.stride}th -> {len(indices)} samples")

    input_sum = input_sqsum = input_count = 0.0
    output_sum = output_sqsum = output_count = 0.0

    start_time = time.time()
    for n, idx in enumerate(indices):
        target, inp, _ = dataset[idx]
        tp_in = box_cox(inp[TP_INPUT_IDX].numpy())
        tp_out = box_cox(target[TP_OUTPUT_IDX].numpy())

        input_sum += tp_in.sum()
        input_sqsum += (tp_in ** 2).sum()
        input_count += tp_in.size

        output_sum += tp_out.sum()
        output_sqsum += (tp_out ** 2).sum()
        output_count += tp_out.size

        if n % 200 == 0:
            elapsed = time.time() - start_time
            print(f"[{n}/{len(indices)}] elapsed={elapsed:.1f}s")

    input_mean = input_sum / input_count
    input_std = np.sqrt(input_sqsum / input_count - input_mean ** 2)
    output_mean = output_sum / output_count
    output_std = np.sqrt(output_sqsum / output_count - output_mean ** 2)

    print()
    print(f"transform_input_means: {{'tp-box_cox_025': {input_mean!r}}}")
    print(f"transform_input_stdevs: {{'tp-box_cox_025': {input_std!r}}}")
    print(f"transform_output_means: {{'tp-box_cox_025': {output_mean!r}}}")
    print(f"transform_output_stdevs: {{'tp-box_cox_025': {output_std!r}}}")


if __name__ == '__main__':
    main()
