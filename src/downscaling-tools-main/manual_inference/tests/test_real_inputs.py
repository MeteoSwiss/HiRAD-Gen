from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from manual_inference.input_data_construction import bundle as bundle_mod


REAL_BUNDLE = Path(
    "/home/ecm5702/hpcperm/data/input_data/o96_o320/idalia/"
    "eefo_o96_0001_date20230826_time0000_mem1_step024h_input_bundle.nc"
)


def test_fill_inputs_from_real_bundle():
    if not REAL_BUNDLE.exists():
        pytest.skip(f"Missing real bundle: {REAL_BUNDLE}")
    try:
        import earthkit.data as _ekd  # noqa: F401
        from anemoi.transform.grids.unstructured import (  # noqa: F401
            UnstructuredGridFieldList,
        )
    except Exception:
        pytest.skip("earthkit/anemoi not available for real bundle test")

    ds = xr.open_dataset(REAL_BUNDLE)
    point_lres = int(ds.sizes["point_lres"])
    point_hres = int(ds.sizes["point_hres"])

    name_to_idx_lres = {}
    expected_lres = []

    if "in_lres_10u" in ds:
        name_to_idx_lres["10u"] = len(name_to_idx_lres)
        expected_lres.append(np.asarray(ds["in_lres_10u"].values, dtype=np.float32))

    if "in_lres_t" in ds and "level" in ds:
        level = int(np.asarray(ds["level"].values)[0])
        name_to_idx_lres[f"t_{level}"] = len(name_to_idx_lres)
        expected_lres.append(
            np.asarray(ds["in_lres_t"].sel(level=level).values, dtype=np.float32)
        )

    if not name_to_idx_lres:
        pytest.skip("Real bundle missing expected LRES variables")

    name_to_idx_hres = {}
    expected_hres = []
    if "in_hres_z" in ds:
        name_to_idx_hres["z"] = len(name_to_idx_hres)
        expected_hres.append(np.asarray(ds["in_hres_z"].values, dtype=np.float32))
    if "in_hres_lsm" in ds:
        name_to_idx_hres["lsm"] = len(name_to_idx_hres)
        expected_hres.append(np.asarray(ds["in_hres_lsm"].values, dtype=np.float32))

    if not name_to_idx_hres:
        pytest.skip("Real bundle missing expected HRES variables")

    x_lres = torch.zeros(
        (1, 1, 1, point_lres, len(name_to_idx_lres)), dtype=torch.float32
    )
    x_hres = torch.zeros(
        (1, 1, 1, point_hres, len(name_to_idx_hres)), dtype=torch.float32
    )

    bundle_mod.fill_inputs_from_bundle(
        REAL_BUNDLE,
        x_lres,
        x_hres,
        name_to_idx_lres,
        name_to_idx_hres,
        device="cpu",
        constant_forcings_npz=None,
    )

    for i, expected in enumerate(expected_lres):
        assert np.allclose(x_lres[0, 0, 0, :, i].numpy(), expected)
    for i, expected in enumerate(expected_hres):
        assert np.allclose(x_hres[0, 0, 0, :, i].numpy(), expected)
