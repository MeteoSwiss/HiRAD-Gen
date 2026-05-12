from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "templates" / "spectra_plot_pdf.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("spectra_plot_pdf", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_curve_summary(variables=("10u", "2t"), n_ell=200):
    ell = list(range(n_ell))
    pred = [float(max(i, 1) ** -2.0) for i in ell]
    truth = [float(max(i, 1) ** -2.1) for i in ell]
    ws = {}
    for v in variables:
        ws[v] = {
            "status": "ok",
            "scopes": {
                "full_field": {
                    "status": "ok",
                    "n_curves": 10,
                    "wavenumbers": ell,
                    "prediction_mean": pred,
                    "truth_mean": truth,
                    "relative_l2_mean_curve": 0.05,
                    "pdf": "/fake/path.pdf",
                },
                "residual": {
                    "status": "ok",
                    "n_curves": 10,
                    "wavenumbers": ell,
                    "prediction_mean": pred,
                    "truth_mean": truth,
                    "relative_l2_mean_curve": 0.08,
                    "pdf": "/fake/residual.pdf",
                },
            },
        }
    return {
        "run_label": "test",
        "predictions_dir": "/fake",
        "spectra_method_used": "healpy",
        "score_wavenumber_min_exclusive": 100.0,
        "weather_states": ws,
    }


def test_build_pdf_proxy_creates_file(tmp_path):
    module = _load_module()
    spectra_dir = tmp_path / "spectra"
    spectra_dir.mkdir()
    (spectra_dir / "spectra_curve_summary.json").write_text(
        json.dumps(_make_curve_summary(["10u", "2t"])), encoding="utf-8"
    )
    out_pdf = tmp_path / "out.pdf"
    n = module.build_pdf(spectra_dir, out_pdf)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert n == 4  # 2 variables x 2 scopes


def test_build_pdf_proxy_skips_bad_status(tmp_path):
    module = _load_module()
    spectra_dir = tmp_path / "spectra"
    spectra_dir.mkdir()
    summary = _make_curve_summary(["10u", "2t"])
    summary["weather_states"]["2t"]["status"] = "error"
    (spectra_dir / "spectra_curve_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    out_pdf = tmp_path / "out.pdf"
    n = module.build_pdf(spectra_dir, out_pdf)
    assert n == 2  # only 10u: 2 scopes


def test_build_pdf_ecmwf_creates_file(tmp_path):
    module = _load_module()
    spectra_dir = tmp_path / "spectra"
    param_dir = spectra_dir / "10u_sfc"
    param_dir.mkdir(parents=True)
    ell = np.arange(512, dtype=float)
    ampl = np.abs(np.random.default_rng(0).standard_normal(512)) + 0.1
    for i in range(3):
        np.save(str(param_dir / f"wvn_20230826_120_{i:03d}.npy"), ell)
        np.save(str(param_dir / f"ampl_20230826_120_{i:03d}.npy"), ampl)
    out_pdf = tmp_path / "ecmwf.pdf"
    n = module.build_pdf(spectra_dir, out_pdf)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
    assert n == 1  # one param dir -> one page


def test_build_pdf_raises_if_no_data(tmp_path):
    module = _load_module()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        module.build_pdf(empty, tmp_path / "out.pdf")
