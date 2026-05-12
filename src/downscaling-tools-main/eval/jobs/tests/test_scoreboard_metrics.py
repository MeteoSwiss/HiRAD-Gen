from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from eval.jobs import generate_enfo_o320_scoreboard as scoreboard
from eval.jobs import scoreboard_metrics as metrics


def test_load_sigma_losses_from_csv_normalizes_float_sigma_labels(tmp_path):
    csv_path = tmp_path / "sample_sigma_eval.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sigma", "loss"])
        writer.writeheader()
        writer.writerow({"sigma": "1.0", "loss": "0.1"})
        writer.writerow({"sigma": "5.0", "loss": "0.2"})
        writer.writerow({"sigma": "10", "loss": "0.3"})

    result = metrics.load_sigma_losses_from_csv(csv_path)

    assert result == {
        "sigma_1": pytest.approx(0.1),
        "sigma_5": pytest.approx(0.2),
        "sigma_10": pytest.approx(0.3),
    }


def test_load_tc_extreme_scores_analysis_anchored(tmp_path):
    """When analysis (OPER) and tail percentiles are present, use analysis-anchored scoring."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "idalia": {
                        "extreme_tail": {
                            "rows": [
                                {
                                    "exp": "OPER_O320_0001",
                                    "mslp_p1": 970.0, "mslp_p01": 965.0, "mslp_min": 960.0,
                                    "wind_p99": 30.0, "wind_p999": 35.0, "wind_max": 40.0,
                                },
                                {
                                    "exp": "ENFO_O320_0001",
                                    "mslp_p1": 980.0, "mslp_p01": 975.0, "mslp_min": 970.0,
                                    "wind_p99": 25.0, "wind_p999": 28.0, "wind_max": 32.0,
                                },
                                {
                                    "exp": "manual_0c446b41_new_o96_o320",
                                    "mslp_p1": 975.0, "mslp_p01": 970.0, "mslp_min": 965.0,
                                    "wind_p99": 28.0, "wind_p999": 32.0, "wind_max": 36.0,
                                },
                            ]
                        }
                    },
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    # Model is between analysis and ENFO — should score > 0.5 but < 1.0
    assert 0.5 < result["idalia"] < 1.0
    # ENFO deviation should be present
    assert "idalia_enfo_dev" in result
    assert result["idalia_enfo_dev"] > 0.0


def test_load_tc_extreme_scores_legacy_fallback(tmp_path):
    """When no OPER row or tail percentiles, fall back to batch-relative normalization."""
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "idalia": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.8, "wind_gt_25_fraction": 0.6},
                            ]
                        }
                    },
                    "franklin": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.2, "wind_gt_25_fraction": 0.2},
                                {"exp": "manual_0c446b41_new_o96_o320", "mslp_980_990_fraction": 0.5, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    },
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(stats_path, run_id="manual_0c446b41_new_o96_o320")

    assert result["idalia"] == pytest.approx(1.0)
    assert result["franklin"] == pytest.approx(1.0)


def test_load_tc_extreme_scores_supports_custom_event_names(tmp_path):
    stats_path = tmp_path / "tc.stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "events": {
                    "humberto": {
                        "extreme_tail": {
                            "rows": [
                                {"exp": "ENFO_O320", "mslp_980_990_fraction": 0.1, "wind_gt_25_fraction": 0.1},
                                {"exp": "manual_95a07500_new_o48_o96", "mslp_980_990_fraction": 0.9, "wind_gt_25_fraction": 0.8},
                            ]
                        }
                    }
                }
            }
        )
    )

    result = metrics.load_tc_extreme_scores_from_json(
        stats_path,
        run_id="manual_95a07500_new_o48_o96",
        event_names=("humberto",),
    )

    assert result == {"humberto": pytest.approx(1.0)}


def test_load_spectra_metrics_falls_back_to_raw_surface_fields(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )

    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        run_curve = np.ones(200, dtype=np.float64)
        ref_curve = np.ones(200, dtype=np.float64)
        run_curve[149] = 2.0
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", ref_curve)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] is not None
    assert result["10v"] is not None
    assert result["2t"] is not None
    # Mean requires at least 3 fields with data
    assert result["mean"] is not None
    assert result["n_curves"] == 1


def test_load_spectra_metrics_ignores_differences_below_high_wavenumber_threshold(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    ref_curve = np.ones(200, dtype=np.float64)
    run_curve = ref_curve.copy()
    run_curve[49] = 5.0  # wavenumber 50 should be ignored by the scoreboard metric

    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", ref_curve)
        np.save(run_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)
        np.save(ref_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] == pytest.approx(0.0)
    assert result["10v"] == pytest.approx(0.0)
    assert result["2t"] == pytest.approx(0.0)
    assert result["mean"] == pytest.approx(0.0)
    assert result["score_wavenumber_min_exclusive"] == pytest.approx(100.0)


def test_load_spectra_metrics_prefers_raw_arrays_over_stale_summary(tmp_path):
    spectra_dir = tmp_path / "spectra_step120_5dates_m10_ecmwf"
    ref_root = tmp_path / "reference"
    spectra_dir.mkdir()
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230826],
                "steps_hours": [120],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )
    (spectra_dir / "spectra_summary.json").write_text(
        json.dumps(
            {
                "method": "ecmwf_mean_curve_reference_l2",
                "10u": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "10v": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "2t": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "msl": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "t_850": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
                "z_500": {"relative_l2_mean_curve": 0.5, "n_pairs": 1},
            }
        )
    )

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    curve = np.ones(200, dtype=np.float64)
    for field_dir in metrics.RAW_FIELD_DIRS.values():
        run_field_dir = spectra_dir / field_dir
        ref_field_dir = ref_root / field_dir
        run_field_dir.mkdir()
        ref_field_dir.mkdir()
        np.save(run_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)
        np.save(ref_field_dir / f"ampl_20230826_120_{field_dir}_1_n1.npy", curve)
        np.save(run_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)
        np.save(ref_field_dir / f"wvn_20230826_120_{field_dir}_1_n1.npy", wvn)

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["mean"] == pytest.approx(0.0)


def test_load_spectra_metrics_falls_back_to_comparison_step_means(tmp_path):
    run_root = tmp_path / "manual_step_mean"
    spectra_dir = run_root / "spectra_proxy10_subset_ecmwf"
    ref_root = run_root / "enfo_o320"
    spectra_dir.mkdir(parents=True)
    ref_root.mkdir()

    (spectra_dir / "staging_summary.json").write_text(
        json.dumps(
            {
                "dates": [20230827],
                "steps_hours": [24],
                "ensemble_members": [1],
                "template_root": str(ref_root),
            }
        )
    )
    (spectra_dir / "spectra_summary.json").write_text(
        json.dumps(
            {
                "weather_states": {
                    "10u": {"status": "missing"},
                    "10v": {"status": "missing"},
                    "2t": {"status": "missing"},
                    "sp": {"status": "missing"},
                    "t_850": {"status": "missing"},
                    "z_500": {"status": "missing"},
                }
            }
        )
    )

    comparison_summary = {
        "base_dir": str(run_root),
        "output_dir": str(spectra_dir),
        "prefer_step": 24,
        "models": [
            {"name": spectra_dir.name, "path": str(spectra_dir), "exists": True, "steps_with_wvn": [24], "chosen_step": 24},
            {"name": ref_root.name, "path": str(ref_root), "exists": True, "steps_with_wvn": [24], "chosen_step": 24},
        ],
        "per_param": {},
    }

    wvn = np.arange(1.0, 201.0, dtype=np.float64)
    ref_curve = np.ones(200, dtype=np.float64)
    run_curve = ref_curve.copy()
    run_curve[149] = 2.0
    expected = metrics.relative_l2_weighted(run_curve, ref_curve, wavenumbers=wvn)

    field_dirs = {
        "10u_sfc": "10u_sfc",
        "10v_sfc": "10v_sfc",
        "2t_sfc": "2t_sfc",
        "sp_sfc": "sp_sfc",
        "t_850": "t_850",
        "z_500": "z_500",
    }
    for field_dir in field_dirs.values():
        (spectra_dir / field_dir).mkdir()
        (ref_root / field_dir).mkdir()
        np.save(spectra_dir / field_dir / f"wvn_20230827_24_{field_dir}_1_n1.npy", wvn)
        np.save(spectra_dir / field_dir / f"ampl_20230827_24_{field_dir}_1_n1.npy", run_curve)
        np.save(ref_root / field_dir / f"wvn_20240201_24_{field_dir}_1_n1.npy", wvn)
        np.save(ref_root / field_dir / f"ampl_20240201_24_{field_dir}_1_n1.npy", ref_curve)
        comparison_summary["per_param"][field_dir] = {
            spectra_dir.name: {"status": "plotted", "step": 24, "files": 1},
            ref_root.name: {"status": "plotted", "step": 24, "files": 1},
        }

    (spectra_dir / "comparison_summary.json").write_text(json.dumps(comparison_summary))

    result = metrics.load_spectra_metrics(spectra_dir)

    assert result["10u"] == pytest.approx(expected)
    assert result["10v"] == pytest.approx(expected)
    assert result["2t"] == pytest.approx(expected)
    assert result["msl"] == pytest.approx(expected)
    assert result["t_850"] == pytest.approx(expected)
    assert result["z_500"] == pytest.approx(expected)
    assert result["mean"] == pytest.approx(expected)
    assert result["count_label"] == "curves"
    assert result["source_path"] == str(spectra_dir / "comparison_summary.json")


def test_build_scoreboard_rows_merges_prefix_related_run_ids():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={
            "manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {
                "weighted_nmse": 0.123,
                "variables": {
                    "10v": {"mean_nmse": 0.2},
                    "2t": {"mean_nmse": 0.01},
                    "msl": {"mean_nmse": 0.05},
                    "sp": {"mean_nmse": 0.02},
                },
            }
        },
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    assert len(rows) == 1
    assert rows[0]["short_id"] == "39991df8"
    assert rows[0]["display_run_id"].startswith("manual_39991df_")
    assert rows[0]["inference"] == "piecewise30"
    assert rows[0]["sigma_1"] == pytest.approx(0.1)
    assert rows[0]["idalia_extreme"] == pytest.approx(0.9)
    assert rows[0]["surface_loss"] == pytest.approx(0.123)
    assert rows[0]["surface_10v"] == pytest.approx(0.2)
    assert rows[0]["surface_2t"] == pytest.approx(0.01)
    assert rows[0]["surface_msl"] == pytest.approx(0.05)
    assert rows[0]["surface_sp"] == pytest.approx(0.02)


def test_load_surface_loss_metrics_uses_truth_std_normalization_fallback(tmp_path):
    summary_path = tmp_path / "surface_loss_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "weighted_surface_mse": 10.0,
                "variables": {
                    "msl": {"mean_mse": 10.0, "normalized_weight": 0.6},
                    "sp": {"mean_mse": 5.0, "normalized_weight": 0.3},
                    "2t": {"mean_mse": 2.0, "normalized_weight": 0.1},
                },
            }
        )
    )

    result = metrics.load_surface_loss_metrics(
        summary_path,
        truth_std_by_variable={"msl": 10.0, "sp": 5.0, "2t": 2.0},
    )

    assert result["weighted_mse"] == pytest.approx(10.0)
    assert result["weighted_nmse"] == pytest.approx(0.6 * 0.1 + 0.3 * 0.2 + 0.1 * 0.5)
    assert [entry["variable"] for entry in result["top_contributors"]] == ["msl", "sp"]
    assert metrics.format_surface_loss_for_scoreboard(result) == "0.1700"


def test_infer_eval_sampler_min_from_run_root_prefers_experiment_config(tmp_path):
    run_root = tmp_path / "manual_piecewise"
    run_root.mkdir()
    (run_root / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "sampling_config_json": json.dumps(
                    {
                        "schedule_type": "experimental_piecewise",
                        "num_steps": 30,
                        "sigma_max": 100000.0,
                        "sigma_min": 0.03,
                    }
                )
            }
        ),
        encoding="utf-8",
    )

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "piecewise30"


def test_infer_eval_sampler_min_from_run_root_falls_back_to_logs(tmp_path):
    run_root = tmp_path / "manual_karras"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "predict25_manual_karras_123.out").write_text(
        "noise_scheduler_params: {'schedule_type': 'karras', 'num_steps': 40, 'sigma_max': 1000.0, 'sigma_min': 0.03, 'rho': 7.0}\n",
        encoding="utf-8",
    )

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "karras40"


def test_infer_eval_sampler_min_from_run_root_falls_back_to_run_id(tmp_path):
    run_root = tmp_path / "manual_83edcda0_multi_o96_o320_20260331_heun40_sigmax1000"
    run_root.mkdir()
    # No EXPERIMENT_CONFIG.yaml and no logs with schedule info

    assert metrics.infer_eval_sampler_min_from_run_root(run_root) == "heun40"


def test_generate_scoreboard_markdown_includes_inference_column():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": 123.0},
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    markdown = scoreboard.generate_scoreboard_markdown(rows)

    assert "| Ckpt | Inference | Run ID |" in markdown
    assert "| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 |" in markdown
    assert "- **Inference**: Schedule-plus-step label inferred from run metadata or prediction logs" in markdown


def test_generate_scoreboard_markdown_prefers_surface_nmse():
    rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": 0.25},
        surface_loss_data={
            "manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {
                "weighted_nmse": 0.17,
                "variables": {
                    "10v": {"mean_nmse": 0.2463},
                    "2t": {"mean_nmse": 0.0082},
                    "msl": {"mean_nmse": 0.0494},
                    "sp": {"mean_nmse": 0.0264},
                },
            }
        },
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )

    markdown = scoreboard.generate_scoreboard_markdown(rows)

    assert "| Ckpt | Inference | Run ID | σ=1 loss | σ=5 loss | σ=10 loss | σ=100 loss | TC Idalia | TC Franklin | ENFO dev | Spectra L2 | Sfc nMSE | 10v nMSE | 2t nMSE | MSLP nMSE | SP nMSE |" in markdown
    assert "| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 | 0.1000 | na | na | na | 0.900 | na | na | 0.2500 | 0.1700 | 0.2463 | 0.0082 | 0.0494 | 0.0264 |" in markdown
    assert "- **Sfc nMSE**: Area-weighted and variable-weighted surface MSE after per-variable truth-std normalization over the fixed Aug 26-30 evaluation contract" in markdown
    assert "- **10v / 2t / MSLP / SP nMSE**: Per-variable truth-std-normalized surface MSE for the named field, using the same fixed evaluation contract as the aggregate surface score" in markdown


def test_load_context_baseline_rows_reads_eefo_o96_input_baseline(tmp_path):
    source_csv = tmp_path / "scoreboard.csv"
    source_csv.write_text(
        "\n".join(
            [
                "tc_rank,label,checkpoint_short,eval_sampler_min,table_group,contract_status,idalia_tc_extreme_score,franklin_tc_extreme_score,spectra_10u_score_vs_reference,spectra_10v_score_vs_reference,spectra_2t_score_vs_reference,spectra_mean_score_vs_reference,spectra_10u_distance,spectra_10v_distance,spectra_2t_distance,spectra_mean_distance,surface_weighted_mse,validation_loss,role,spectra_coverage,spectra_n_curves,surface_loss_source,note,dossier",
                "-,enfo_o320,na,na,context,eligible,0.859539,na,1.000000,1.000000,1.000000,1.000000,0.000000,0.000000,0.000000,0.000000,na,na,truth-reference,reference by definition,na,na,Reference baseline.,/tmp/enfo.md",
                "-,eefo_o96,na,na,context,eligible,0.000000,na,na,na,na,na,na,na,na,na,na,na,input-baseline,missing,na,na,Context baseline (off-grid input).,/tmp/eefo.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = scoreboard.load_context_baseline_rows(source_csv)

    assert len(rows) == 1
    assert rows[0]["short_id"] == "x_interp"
    assert rows[0]["display_run_id"] == "eefo_o96"
    assert rows[0]["idalia_extreme"] == pytest.approx(0.0)
    assert math.isnan(rows[0]["sigma_1"])
    assert math.isnan(rows[0]["spectra_l2"])
    assert math.isnan(rows[0]["surface_loss"])


def test_load_context_baseline_rows_overrides_eefo_surface_with_real_xinterp_metrics(tmp_path):
    source_csv = tmp_path / "scoreboard.csv"
    source_csv.write_text(
        "\n".join(
            [
                "tc_rank,label,checkpoint_short,eval_sampler_min,table_group,contract_status,idalia_tc_extreme_score,franklin_tc_extreme_score,spectra_10u_score_vs_reference,spectra_10v_score_vs_reference,spectra_2t_score_vs_reference,spectra_mean_score_vs_reference,spectra_10u_distance,spectra_10v_distance,spectra_2t_distance,spectra_mean_distance,surface_weighted_mse,validation_loss,role,spectra_coverage,spectra_n_curves,surface_loss_source,note,dossier",
                "-,eefo_o96,na,na,context,eligible,0.000000,na,na,na,na,na,na,na,na,na,na,na,input-baseline,missing,na,na,Context baseline (off-grid input).,/tmp/eefo.md",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = scoreboard.load_context_baseline_rows(
        source_csv,
        context_surface_metrics={
            "eefo_o96": {
                "weighted_nmse": 0.1026,
                "variables": {
                    "10v": {"mean_nmse": 0.2},
                    "2t": {"mean_nmse": 0.01},
                    "msl": {"mean_nmse": 0.04},
                    "sp": {"mean_nmse": 0.03},
                },
            }
        },
    )

    assert rows[0]["surface_loss"] == pytest.approx(0.1026)
    assert rows[0]["surface_loss_text"] == "0.1026"
    assert rows[0]["surface_10v"] == pytest.approx(0.2)
    assert rows[0]["surface_2t"] == pytest.approx(0.01)
    assert rows[0]["surface_msl"] == pytest.approx(0.04)
    assert rows[0]["surface_sp"] == pytest.approx(0.03)


def test_generate_scoreboard_markdown_appends_eefo_o96_context_row():
    experiment_rows = scoreboard.build_scoreboard_rows(
        sigma_data={"39991df81216460fb7f3bd048df733c3": {"sigma_1": 0.1}},
        tc_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": {"idalia_extreme": 0.9}},
        spectra_data={},
        surface_loss_data={},
        inference_data={"manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100": "piecewise30"},
    )
    experiment_rows.append(
        {
            "row_key": "eefo_o96",
            "display_run_id": "eefo_o96",
            "short_id": "x_interp",
            "inference": "na",
            "row_group": "context",
            "sigma_1": float("nan"),
            "sigma_5": float("nan"),
            "sigma_10": float("nan"),
            "sigma_100": float("nan"),
            "idalia_extreme": 0.0,
            "franklin_extreme": float("nan"),
            "enfo_deviation": float("nan"),
            "spectra_l2": float("nan"),
            "surface_loss": float("nan"),
            "surface_10v": float("nan"),
            "surface_2t": float("nan"),
            "surface_msl": float("nan"),
            "surface_sp": float("nan"),
        }
    )

    markdown = scoreboard.generate_scoreboard_markdown(experiment_rows)

    assert "| x_interp | na | eefo_o96 | na | na | na | na | 0.000 | na | na | na | na | na | na | na | na |" in markdown
    assert markdown.index("| 39991df8 | piecewise30 | manual_39991df_new_o96_o320_20260317_piecewise30_h10_l20_sigma100 |") < markdown.index("| x_interp | na | eefo_o96 |")
    assert "- **Context baseline rows**: Curated comparison rows appended after experiment runs; `x_interp` is sourced from the docs-side `eefo_o96` input baseline" in markdown


def test_choose_xinterp_context_predictions_dir_prefers_matching_o96_o320_contract(tmp_path, monkeypatch):
    other_run = tmp_path / "manual_181be03e_new_o320_o1280_20260421_manual_eval"
    other_predictions = other_run / "predictions"
    other_predictions.mkdir(parents=True)
    (other_predictions / "predictions_20230827_step024.nc").write_text("", encoding="utf-8")
    (other_run / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "lane": "o320_o1280",
                "source": {
                    "bundle_scope": {
                        "dates": [20230827, 20230828, 20230829, 20230830],
                        "steps_hours": [24, 48, 72],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    matching_run = tmp_path / "manual_59e4_300k"
    matching_predictions = matching_run / "predictions"
    matching_predictions.mkdir(parents=True)
    (matching_predictions / "predictions_20230826_step024.nc").write_text("", encoding="utf-8")
    (matching_run / "EXPERIMENT_CONFIG.yaml").write_text(
        json.dumps(
            {
                "lane": "o96_o320",
                "source": {
                    "bundle_scope": {
                        "dates": [20230826, 20230827, 20230828, 20230829, 20230830],
                        "steps_hours": [24, 48, 72, 96, 120],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        scoreboard,
        "_predictions_support_xinterp",
        lambda path: path == matching_predictions,
    )

    chosen = scoreboard.choose_xinterp_context_predictions_dir(tmp_path)

    assert chosen == matching_predictions


def test_finite_positive_mask_adaptive_threshold_low_lmax():
    """When max wavenumber <= 100, threshold drops to max_wvn/3."""
    wvn = np.arange(0, 96, dtype=np.float64)  # lmax=95
    arr = np.ones_like(wvn) * 10.0
    mask = metrics.finite_positive_mask(arr, wavenumbers=wvn)
    # Effective threshold = 95/3 ≈ 31.67, so wavenumbers > 31.67 should pass
    assert mask.sum() > 0, "Should score wavenumbers when lmax < 100"
    assert mask[0] is np.False_, "Wavenumber 0 should be excluded"
    assert mask[32] is np.True_, "Wavenumber 32 should be included"


def test_finite_positive_mask_standard_threshold_high_lmax():
    """When max wavenumber > 100, threshold stays at 100."""
    wvn = np.arange(0, 321, dtype=np.float64)  # lmax=320
    arr = np.ones_like(wvn) * 10.0
    mask = metrics.finite_positive_mask(arr, wavenumbers=wvn)
    # Threshold stays at 100
    assert mask[100] is np.False_, "Wavenumber 100 should be excluded (> not >=)"
    assert mask[101] is np.True_, "Wavenumber 101 should be included"
    assert mask[50] is np.False_, "Wavenumber 50 should be excluded"


def test_build_run_scoreboard_metrics_sigma_fallback_to_run_root(tmp_path):
    """When SIGMA_RUN_ID is blank, sigma loads from <run_root>/sigma_eval_table.csv."""
    run_id = "manual_abc123_new_o48_o96_20260422_full_eval"
    run_root = tmp_path / run_id
    run_root.mkdir()

    # Write sigma CSV to the run root (not scoreboards/sigma/)
    sigma_csv = run_root / "sigma_eval_table.csv"
    with sigma_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "loss"])
        writer.writeheader()
        writer.writerow({"sigma": "1.0", "loss": "0.098"})
        writer.writerow({"sigma": "5.0", "loss": "0.139"})
        writer.writerow({"sigma": "10.0", "loss": "0.162"})
        writer.writerow({"sigma": "100.0", "loss": "0.211"})

    # Empty TC/spectra/surface stubs
    tc_path = run_root / "tc_stats.json"
    tc_path.write_text("{}", encoding="utf-8")
    spectra_dir = run_root / "spectra"
    spectra_dir.mkdir()
    surface_path = run_root / "surface_loss.json"
    surface_path.write_text("{}", encoding="utf-8")

    result = metrics.build_run_scoreboard_metrics(
        run_id=run_id,
        output_root=tmp_path,
        sigma_run_id="",
        tc_stats_path=tc_path,
        spectra_dir=spectra_dir,
        surface_json_path=surface_path,
    )

    assert result["sigma_losses"]["sigma_1"] == pytest.approx(0.098)
    assert result["sigma_losses"]["sigma_5"] == pytest.approx(0.139)
    assert result["sigma_losses"]["sigma_10"] == pytest.approx(0.162)
    assert result["sigma_losses"]["sigma_100"] == pytest.approx(0.211)


def test_rescore_from_curve_summary_low_lmax(tmp_path):
    """_rescore_from_curve_summary rescores when lmax < 100."""
    spectra_dir = tmp_path / "spectra_o48_o96"
    spectra_dir.mkdir()

    # Build a curve summary with lmax=95 and identical pred/truth means
    wvn = list(range(96))  # 0..95
    pred_mean = [float(i + 1) for i in range(96)]
    truth_mean = [float(i + 1) for i in range(96)]

    weather_states = {}
    for field in ["10u", "10v", "2t", "msl", "t_850", "z_500"]:
        weather_states[field] = {
            "status": "ok",
            "scopes": {
                "residual": {
                    "status": "ok",
                    "n_curves": 25,
                    "wavenumbers": wvn,
                    "prediction_mean": pred_mean,
                    "truth_mean": truth_mean,
                }
            },
        }

    (spectra_dir / "spectra_curve_summary.json").write_text(
        json.dumps({
            "score_wavenumber_min_exclusive": 100.0,
            "weather_states": weather_states,
        }),
        encoding="utf-8",
    )

    result = metrics._rescore_from_curve_summary(spectra_dir)

    assert result["mean"] is not None
    assert result["mean"] == pytest.approx(0.0)
    assert result["coverage"] == "25 curves"
    for field in ["10u", "10v", "2t", "msl", "t_850", "z_500"]:
        assert result[field] == pytest.approx(0.0)
