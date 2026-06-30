import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_TNPDIR = Path(__file__).parents[4] / "deps" / "gridded-tnp"
if _TNPDIR.exists() and str(_TNPDIR) not in sys.path:
    sys.path.insert(0, str(_TNPDIR))

from tnp.data.base import OOTGBatch

from oat.baselines.coord_utils import lv95_to_latlon, nwp_latlon_grid
from oat.baselines.ootg_generator import (
    OATOOTGDataGenerator,
    compute_norm_stats,
    make_ootg_generator,
)


# --- helpers -----------------------------------------------------------

def _make_mock_dataset(n_stations=20, n_obs=1, n_nwp=2, H=20, W=20):
    dist_map = (
        {"temperature": "normal"}
        if n_obs == 1
        else {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    )
    cfg = SimpleNamespace(
        dataset_type="cosmo_peakweather",
        nwp_vars=["T_2M", "T_G"][:n_nwp],
        topo_vars=["DEM"],
        obs_vars=list(dist_map.keys()),
        obs_variable_to_distribution=dist_map,
        ref_nwp_path="mock_path",
    )
    obs = torch.zeros(n_stations, n_obs)
    obs[:, 0] = torch.randn(n_stations) * 5.0 + 280.0  # temperature (Kelvin)
    if n_obs == 2:
        obs[:, 1] = torch.rand(n_stations) * 5.0       # precipitation (non-negative)
    mask = torch.ones(n_stations, n_obs)
    mask[0] = 0.0
    coords = torch.rand(n_stations, 2) * 1e5 + torch.tensor([2.5e6, 1.1e6])
    global_field = torch.randn(n_nwp + 1, H, W)
    sample = {
        "global_data": {"global_field": global_field},
        "observation_data": {"obs": obs, "mask_obs": mask, "coords": coords},
    }
    ds = MagicMock()
    ds.config = cfg
    ds.__len__ = MagicMock(return_value=100)
    ds.__getitem__ = MagicMock(return_value=sample)
    return ds


def _xc_grid():
    ref_x = np.linspace(2255500.0, 2964500.0, 20)
    ref_y = np.linspace(840500.0, 1479500.0, 20)
    return nwp_latlon_grid(ref_x, ref_y, stride=2)  # (10, 10, 2)


def _make_generator(n_stations=20, n_obs=1):
    ds = _make_mock_dataset(n_stations, n_obs=n_obs)
    obs_mean = torch.full((n_obs,), 280.0)
    obs_std = torch.full((n_obs,), 5.0)
    nwp_mean, nwp_std = torch.zeros(2), torch.ones(2)
    dist_names = ["normal"] if n_obs == 1 else ["normal", "zero_inflated_lognormal"]
    return OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=obs_mean, obs_std=obs_std,
        nwp_mean=nwp_mean, nwp_std=nwp_std,
        xc_grid_latlon=_xc_grid(),
        coord_to_latlon=lv95_to_latlon,
        dist_names=dist_names,
        nwp_stride=2,
        samples_per_epoch=10,
    )


# --- tests -------------------------------------------------------------

def test_generate_batch_returns_ootg_batch():
    assert isinstance(_make_generator().generate_batch(), OOTGBatch)


def test_batch_coordinate_dims():
    batch = _make_generator().generate_batch()
    assert batch.xc.shape[0] == 1 and batch.xc.shape[2] == 2
    assert batch.yc.shape[2] == 1
    assert batch.xt.shape[2] == 2 and batch.yt.shape[2] == 1


def test_batch_coordinate_dims_multivariate():
    batch = _make_generator(n_obs=2).generate_batch()
    assert batch.yc.shape[2] == 2 and batch.yt.shape[2] == 2


def test_batch_grid_shapes():
    batch = _make_generator().generate_batch()
    assert batch.xc_grid.shape == (1, 10, 10, 2)
    assert batch.yc_grid.shape == (1, 10, 10, 2)


def test_context_plus_target_equals_valid_stations():
    batch = _make_generator(n_stations=20).generate_batch()
    assert batch.xc.shape[1] + batch.xt.shape[1] == 19


def test_ziln_variable_not_normalized():
    gen = _make_generator(n_obs=2)
    batch = gen.generate_batch()
    # precipitation (index 1) should remain non-negative (original scale)
    assert (batch.yc[0, :, 1] >= 0).all(), "precipitation context values must be non-negative"
    assert (batch.yt[0, :, 1] >= 0).all(), "precipitation target values must be non-negative"


def test_normal_variable_is_normalized():
    gen = _make_generator(n_obs=1)
    batch = gen.generate_batch()
    # temperature is normalized around 0 — abs values should be much smaller than raw K values
    assert batch.yc[0, :, 0].abs().max() < 10.0


def test_latlon_range():
    batch = _make_generator().generate_batch()
    xc = batch.xc[0]
    assert xc[:, 0].min() > 45.0 and xc[:, 0].max() < 49.0
    assert xc[:, 1].min() > 5.0 and xc[:, 1].max() < 11.0


def test_time_indices_restricts_samples():
    ds = _make_mock_dataset()
    obs_mean, obs_std = torch.tensor([280.0]), torch.tensor([5.0])
    nwp_mean, nwp_std = torch.zeros(2), torch.ones(2)
    gen = OATOOTGDataGenerator(
        dataset=ds, obs_mean=obs_mean, obs_std=obs_std,
        nwp_mean=nwp_mean, nwp_std=nwp_std,
        xc_grid_latlon=_xc_grid(),
        coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"],
        nwp_stride=2, samples_per_epoch=10,
        time_indices=np.array([0, 1, 2]),
    )
    gen.generate_batch()
    assert ds.__getitem__.call_args[0][0] in [0, 1, 2]


def test_compute_norm_stats_shapes():
    ds = _make_mock_dataset()
    om, os_, nm, ns = compute_norm_stats(ds, n_samples=10)
    assert om.shape == (1,) and os_.shape == (1,)
    assert nm.shape == (2,) and ns.shape == (2,)
    assert os_.item() > 0 and (ns > 0).all()


def test_compute_norm_stats_multivariate():
    ds = _make_mock_dataset(n_obs=2)
    om, os_, nm, ns = compute_norm_stats(ds, n_samples=10)
    assert om.shape == (2,) and os_.shape == (2,)


def test_make_ootg_generator_cosmo():
    ds = _make_mock_dataset()
    obs_mean, obs_std = torch.tensor([280.0]), torch.tensor([5.0])
    nwp_mean, nwp_std = torch.zeros(2), torch.ones(2)
    from unittest.mock import patch
    mock_ref = MagicMock()
    mock_ref.x.values = np.linspace(2255500.0, 2964500.0, 20)
    mock_ref.y.values = np.linspace(840500.0, 1479500.0, 20)
    with patch("oat.baselines.ootg_generator.xr") as mock_xr:
        mock_xr.open_zarr.return_value = mock_ref
        gen = make_ootg_generator(
            ds, obs_mean, obs_std, nwp_mean, nwp_std, nwp_stride=2, samples_per_epoch=5
        )
    assert isinstance(gen, OATOOTGDataGenerator)
    assert gen.dist_names == ["normal"]
