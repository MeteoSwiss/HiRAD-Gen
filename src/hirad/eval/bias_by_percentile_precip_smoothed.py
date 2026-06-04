"""Smoothed precipitation bias / MAE / spread by percentile plots.

Applies the same Gaussian low-pass filter to target and prediction fields before
the percentile histograms are accumulated, mitigating double-penalty effects in
the precipitation tails.
"""
from hirad.eval.bias_by_percentile_common import run_bias_by_percentile
from hirad.eval.bias_by_percentile_precip import SPEC
from hirad.eval.eval_utils import parse_eval_cli


SMOOTHING_SIGMA_KM = 20.0
GRID_RES_KM = 1.0


def main(cfg: dict) -> None:
    cfg = dict(cfg)
    cfg['smoothing_sigma_km'] = SMOOTHING_SIGMA_KM
    cfg.setdefault('grid_res_km', GRID_RES_KM)
    run_bias_by_percentile(cfg, SPEC)


if __name__ == '__main__':
    main(parse_eval_cli())