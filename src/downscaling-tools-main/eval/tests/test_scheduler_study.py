from __future__ import annotations

import json

import torch

from eval._backends.sigma_evaluator import scheduler_study as mod


def _profiles() -> list[mod.CheckpointProfile]:
    return [
        mod.CheckpointProfile(
            checkpoint_id="61cf18112f9f4e5da192ae930b40aa79",
            checkpoint_short="61cf1811",
            checkpoint_path="/tmp/61cf1811.ckpt",
            baseline_slug="manual-61cf1811-new-piecewise30-h10-l20-sigma100",
        ),
        mod.CheckpointProfile(
            checkpoint_id="cfec83a3cd0644778e2bfcbacfa9f4fc",
            checkpoint_short="cfec83a3",
            checkpoint_path="/tmp/cfec83a3.ckpt",
            baseline_slug="manual-cfec83a3-new-piecewise30-h10-l20-sigma100",
        ),
    ]


def test_materialize_sigma_schedule_uses_torch_dtype(monkeypatch):
    captured = {}

    class _Scheduler:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = dict(kwargs)

        def get_schedule(self, device=None, dtype_compute=None, **_kwargs):
            captured["device"] = device
            captured["dtype_compute"] = dtype_compute
            return torch.tensor([captured["init_kwargs"]["sigma_max"], 0.0], dtype=torch.float64)

    monkeypatch.setattr(mod, "_get_noise_scheduler_classes", lambda: {"experimental_piecewise": _Scheduler})

    schedule = mod.materialize_sigma_schedule(
        {
            "schedule_type": "experimental_piecewise",
            "sigma_max": 10000.0,
            "sigma_min": 0.03,
            "num_steps": 23,
            "sigma_transition": 10.0,
            "num_steps_high": 3,
            "num_steps_low": 20,
            "high_schedule_type": "exponential",
            "low_schedule_type": "karras",
            "rho": 7.0,
        }
    )

    assert captured["dtype_compute"] is torch.float64
    assert captured["init_kwargs"]["num_steps_high"] == 3
    assert schedule == [10000.0, 0.0]


def test_build_stage0_manifest_includes_controls_and_safe_run_ids(monkeypatch):
    monkeypatch.setattr(
        mod,
        "materialize_sigma_schedule",
        lambda extra_args: [
            float(extra_args["sigma_max"]),
            float(extra_args.get("sigma_transition", extra_args["sigma_min"])),
            0.0,
        ],
    )

    manifest = mod.build_stage0_manifest(
        _profiles(),
        run_date="20260328",
        sigma_transition=10.0,
        sigma_max_values=(1000.0,),
        num_steps_high_values=(3,),
        num_steps_low=20,
    )

    assert manifest["stage"] == "stage0"
    assert manifest["candidate_count"] == 8
    candidate_keys = {candidate["candidate_key"] for candidate in manifest["candidates"]}
    assert "stage0_piecewise_smax1k_t10_h3_l20" in candidate_keys
    assert "control_piecewise30_reference" in candidate_keys
    assert "control_karras40_sigmax1000" in candidate_keys
    assert "control_karras80_sigmax100000" in candidate_keys

    stage0_candidate = next(
        candidate for candidate in manifest["candidates"] if candidate["candidate_key"] == "stage0_piecewise_smax1k_t10_h3_l20"
    )
    assert stage0_candidate["run_id"] == "manual_61cf1811_new_o96_o320_20260328_stage0_piecewise_smax1k_t10_h3_l20"
    assert stage0_candidate["total_steps"] == 23
    assert stage0_candidate["sigma_schedule"][-1] == 0.0


def test_build_stage1_manifest_can_sweep_high_and_low_steps(monkeypatch):
    monkeypatch.setattr(mod, "materialize_sigma_schedule", lambda extra_args: [float(extra_args["sigma_max"]), 0.0])
    best_candidate = {
        "study": "o96_o320_piecewise_scheduler_search",
        "stage": "stage0",
        "checkpoint_id": "61cf18112f9f4e5da192ae930b40aa79",
        "checkpoint_short": "61cf1811",
        "checkpoint_path": "/tmp/61cf1811.ckpt",
        "stack_flavor": "new",
        "lane": "o96_o320",
        "baseline_slug": "manual-61cf1811-new-piecewise30-h10-l20-sigma100",
        "candidate_key": "winner",
        "candidate_label": "winner",
        "family": "search",
        "run_id": "manual_61cf1811_new_o96_o320_20260328_winner",
        "total_steps": 25,
        "extra_args": {
            "schedule_type": "experimental_piecewise",
            "num_steps": 25,
            "sigma_max": 1000.0,
            "sigma_transition": 10.0,
            "sigma_min": 0.03,
            "num_steps_high": 5,
            "num_steps_low": 20,
            "rho": 7.0,
            "sampler": "heun",
            "S_churn": 2.5,
            "S_min": 0.75,
            "S_max": 1000.0,
            "S_noise": 1.05,
        },
        "sigma_schedule": [1000.0, 0.0],
    }

    manifest = mod.build_stage1_manifest(
        _profiles()[:1],
        best_candidate,
        run_date="20260330",
        transition_values=(5.0, 7.0, 10.0),
        ensure_transition_controls=False,
        num_steps_high_values=(5, 7, 9),
        num_steps_low_values=(13, 16, 19),
        include_controls=False,
    )

    combos = {
        (
            candidate["extra_args"]["sigma_transition"],
            candidate["extra_args"]["num_steps_high"],
            candidate["extra_args"]["num_steps_low"],
        )
        for candidate in manifest["candidates"]
    }
    assert manifest["candidate_count"] == 27
    assert len(combos) == 27
    assert (5.0, 5, 13) in combos
    assert (7.0, 7, 16) in combos
    assert (10.0, 9, 19) in combos
    assert {candidate["extra_args"]["sigma_max"] for candidate in manifest["candidates"]} == {1000.0}


def test_build_stage2_manifest_uses_local_neighborhood(monkeypatch):
    monkeypatch.setattr(mod, "materialize_sigma_schedule", lambda extra_args: [1.0, 0.0])
    best_candidate = {
        "study": "o96_o320_piecewise_scheduler_search",
        "stage": "stage1",
        "checkpoint_id": "61cf18112f9f4e5da192ae930b40aa79",
        "checkpoint_short": "61cf1811",
        "checkpoint_path": "/tmp/61cf1811.ckpt",
        "stack_flavor": "new",
        "lane": "o96_o320",
        "baseline_slug": "manual-61cf1811-new-piecewise30-h10-l20-sigma100",
        "candidate_key": "winner",
        "candidate_label": "winner",
        "family": "search",
        "run_id": "manual_61cf1811_new_o96_o320_20260328_winner",
        "total_steps": 21,
        "extra_args": {
            "schedule_type": "experimental_piecewise",
            "num_steps": 21,
            "sigma_max": 10000.0,
            "sigma_transition": 3.0,
            "sigma_min": 0.03,
            "num_steps_high": 5,
            "num_steps_low": 16,
            "rho": 7.0,
            "sampler": "heun",
            "S_churn": 2.5,
            "S_min": 0.75,
            "S_max": 10000.0,
            "S_noise": 1.05,
        },
        "sigma_schedule": [1.0, 0.0],
    }

    manifest = mod.build_stage2_manifest(
        _profiles()[:1],
        best_candidate,
        run_date="20260328",
        sigma_max_values=(1000.0, 10000.0, 100000.0),
        num_steps_high_values=(3, 5, 7),
        include_controls=False,
    )

    combos = {
        (candidate["extra_args"]["sigma_max"], candidate["extra_args"]["num_steps_high"])
        for candidate in manifest["candidates"]
    }
    assert combos == {
        (1000.0, 3),
        (1000.0, 5),
        (1000.0, 7),
        (10000.0, 3),
        (10000.0, 5),
        (10000.0, 7),
        (100000.0, 3),
        (100000.0, 5),
        (100000.0, 7),
    }


def test_analyze_transition_prior_keeps_control_band_and_seed_order():
    ell = list(range(1, 101))
    low_res_power = [1.0] * 100
    high_res_power = [1.0] * 79 + [5.0] * 21
    result = mod.analyze_transition_prior(
        ell=ell,
        low_res_power=low_res_power,
        high_res_power=high_res_power,
    )

    assert result["departure_ell"] >= 79.0
    assert result["seed_transition"] == 1.0
    assert result["candidate_transitions"] == [1.0, 3.0, 10.0, 30.0, 100.0]
    assert result["priority_order"][0] == 1.0


def test_rank_scheduler_results_respects_guardrail_and_step_tiebreak():
    payload = {
        "baseline_metrics": {
            "61cf1811": {
                "idalia_tc": 0.955691,
                "spectra_mean": 0.967218,
                "spectra_10u": 0.962478,
                "spectra_10v": 0.951560,
                "spectra_2t": 0.987614,
            },
            "cfec83a3": {
                "idalia_tc": 0.909171,
                "spectra_mean": 0.979096,
                "spectra_10u": 0.977953,
                "spectra_10v": 0.972759,
                "spectra_2t": 0.986575,
            },
        },
        "results": [
            {
                "candidate_key": "candidate_a",
                "candidate_label": "candidate_a",
                "checkpoint_short": "61cf1811",
                "total_steps": 23,
                "idalia_tc": 0.965691,
                "spectra_mean": 0.965000,
                "spectra_10u": 0.960000,
                "spectra_10v": 0.950000,
                "spectra_2t": 0.986000,
            },
            {
                "candidate_key": "candidate_a",
                "candidate_label": "candidate_a",
                "checkpoint_short": "cfec83a3",
                "total_steps": 23,
                "idalia_tc": 0.919171,
                "spectra_mean": 0.977000,
                "spectra_10u": 0.976000,
                "spectra_10v": 0.972000,
                "spectra_2t": 0.985500,
            },
            {
                "candidate_key": "candidate_b",
                "candidate_label": "candidate_b",
                "checkpoint_short": "61cf1811",
                "total_steps": 27,
                "idalia_tc": 0.965691,
                "spectra_mean": 0.965000,
                "spectra_10u": 0.960000,
                "spectra_10v": 0.950000,
                "spectra_2t": 0.986000,
            },
            {
                "candidate_key": "candidate_b",
                "candidate_label": "candidate_b",
                "checkpoint_short": "cfec83a3",
                "total_steps": 27,
                "idalia_tc": 0.919171,
                "spectra_mean": 0.977000,
                "spectra_10u": 0.976000,
                "spectra_10v": 0.972000,
                "spectra_2t": 0.985500,
            },
            {
                "candidate_key": "candidate_bad",
                "candidate_label": "candidate_bad",
                "checkpoint_short": "61cf1811",
                "total_steps": 23,
                "idalia_tc": 1.005691,
                "spectra_mean": 0.940000,
                "spectra_10u": 0.930000,
                "spectra_10v": 0.920000,
                "spectra_2t": 0.950000,
            },
            {
                "candidate_key": "candidate_bad",
                "candidate_label": "candidate_bad",
                "checkpoint_short": "cfec83a3",
                "total_steps": 23,
                "idalia_tc": 0.959171,
                "spectra_mean": 0.930000,
                "spectra_10u": 0.920000,
                "spectra_10v": 0.910000,
                "spectra_2t": 0.940000,
            },
        ],
    }

    ranked = mod.rank_scheduler_results(payload)

    assert ranked[0]["candidate_key"] == "candidate_a"
    assert ranked[0]["guardrail_pass"] is True
    assert ranked[1]["candidate_key"] == "candidate_b"
    assert ranked[1]["guardrail_pass"] is True
    assert ranked[2]["candidate_key"] == "candidate_bad"
    assert ranked[2]["guardrail_pass"] is False


def test_build_ecmwf_stage_scoreboard_uses_control_relative_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_load_ecmwf_spectra_metrics",
        lambda spectra_dir, reference_root=None: {
            "mean": 0.0 if spectra_dir.parent.name.endswith("control_piecewise30_reference") else 0.01234,
            "10u": 0.0 if spectra_dir.parent.name.endswith("control_piecewise30_reference") else 0.01300,
            "10v": 0.0 if spectra_dir.parent.name.endswith("control_piecewise30_reference") else 0.01400,
            "2t": 0.0 if spectra_dir.parent.name.endswith("control_piecewise30_reference") else 0.01000,
            "coverage": "10u: 4d/1m/3s/10c; 10v: 4d/1m/3s/10c; 2t: 4d/1m/3s/10c",
            "n_curves": 10,
        },
    )

    manifest = mod.build_stage0_manifest(
        _profiles()[:1],
        run_date="20260328",
        sigma_transition=10.0,
        sigma_max_values=(1000.0,),
        num_steps_high_values=(3,),
        num_steps_low=20,
    )

    control_root = tmp_path / "manual_61cf1811_new_o96_o320_20260328_control_piecewise30_reference"
    search_root = tmp_path / "manual_61cf1811_new_o96_o320_20260328_stage0_piecewise_smax1k_t10_h3_l20"
    for run_root, wind_max, mslp_min in ((control_root, 31.5, 976.0), (search_root, 33.0, 972.5)):
        (run_root / "spectra_ecmwf").mkdir(parents=True)
        (run_root / "proxy_tc_compare.json").write_text(
            json.dumps(
                {
                    "events": {
                        "idalia": {
                            "status": "FAIL",
                            "max_deviation": 1.25,
                            "aggregate": {
                                "mslp_min": {"proxy_extreme": mslp_min},
                                "wind_max": {"proxy_extreme": wind_max},
                                "mslp_980_990_count": {"proxy": 10},
                                "wind_gt_25_count": {"proxy": 20},
                            },
                        },
                        "franklin": {
                            "status": "FAIL",
                            "max_deviation": 0.75,
                            "aggregate": {
                                "mslp_min": {"proxy_extreme": mslp_min + 2.0},
                                "wind_max": {"proxy_extreme": wind_max - 1.0},
                                "mslp_980_990_count": {"proxy": 7},
                                "wind_gt_25_count": {"proxy": 9},
                            },
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    ledger = [
        {
            "run_id": "manual_61cf1811_new_o96_o320_20260328_control_piecewise30_reference",
            "candidate_key": "control_piecewise30_reference",
            "checkpoint_short": "61cf1811",
            "family": "control",
            "total_steps": "30",
            "predict_job_id": "1",
            "eval_job_id": "2",
            "run_root": str(control_root),
        },
        {
            "run_id": "manual_61cf1811_new_o96_o320_20260328_stage0_piecewise_smax1k_t10_h3_l20",
            "candidate_key": "stage0_piecewise_smax1k_t10_h3_l20",
            "checkpoint_short": "61cf1811",
            "family": "search",
            "total_steps": "23",
            "predict_job_id": "3",
            "eval_job_id": "4",
            "run_root": str(search_root),
        },
    ]

    payload = mod.build_ecmwf_stage_scoreboard(manifest, ledger)

    assert payload["row_count"] == 2
    search_row = next(row for row in payload["rows"] if row["candidate_key"] == "stage0_piecewise_smax1k_t10_h3_l20")
    assert search_row["spectra_vs_control_mean"] == 0.01234
    assert search_row["idalia_wind_max_delta_vs_control"] == 1.5
    assert search_row["idalia_mslp_min_delta_vs_control"] == -3.5
    assert search_row["franklin_wind_max_delta_vs_control"] == 1.5
    assert search_row["spectra_vs_control_coverage"].startswith("10u:")
