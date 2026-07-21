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
    OOTGTopoBatch,
    compute_norm_stats,
    make_ootg_generator,
)
from oat.data_module.transform import StandardNormalizeTransform


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
    nwp_mean, nwp_std = torch.zeros(3), torch.ones(3)
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


C_TOPO, HP, WP = 3, 16, 16


def _add_topo_to_dataset(ds, n_stations, c_topo=C_TOPO, hp=HP, wp=WP):
    """Attach a topo patch tensor to the mock dataset's returned sample."""
    sample = ds.__getitem__.return_value
    sample["observation_data"]["topo"] = torch.randn(n_stations, c_topo, hp, wp)
    return ds


def _topo_transform(c_topo=C_TOPO):
    return StandardNormalizeTransform(
        mean=torch.zeros(c_topo), std=torch.ones(c_topo), channel_dim=-3
    )


def _make_topo_generator(n_stations=20, n_obs=1):
    ds = _add_topo_to_dataset(_make_mock_dataset(n_stations, n_obs=n_obs), n_stations)
    obs_mean = torch.full((n_obs,), 280.0)
    obs_std = torch.full((n_obs,), 5.0)
    nwp_mean, nwp_std = torch.zeros(3), torch.ones(3)
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
        topo_transform=_topo_transform(),
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
    assert batch.yc_grid.shape == (1, 10, 10, 3)


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
    nwp_mean, nwp_std = torch.zeros(3), torch.ones(3)
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


def test_eligible_station_indices_restricts_pool():
    ds = _make_mock_dataset(n_stations=20)
    ds.__getitem__.return_value["observation_data"]["mask_obs"][:] = 1.0
    eligible = np.array([0, 1, 2, 3, 4])
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        eligible_station_indices=eligible,
    )
    for _ in range(20):
        batch = gen.generate_batch()
        assert batch.xc.shape[1] + batch.xt.shape[1] <= len(eligible)


def test_context_target_station_indices_fixed_roles():
    ds = _make_mock_dataset(n_stations=20)
    ds.__getitem__.return_value["observation_data"]["mask_obs"][:] = 1.0
    context_idx = np.array([0, 1, 2, 3, 4])
    target_idx = np.array([10, 11, 12])
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        context_station_indices=context_idx, target_station_indices=target_idx,
    )
    for _ in range(10):
        batch = gen.generate_batch()
        assert batch.xc.shape[1] == len(context_idx)
        assert batch.xt.shape[1] == len(target_idx)


def test_context_without_target_raises():
    ds = _make_mock_dataset()
    with pytest.raises(ValueError):
        OATOOTGDataGenerator(
            dataset=ds,
            obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
            nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
            xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
            dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
            context_station_indices=np.array([0, 1]),
        )


def test_conflicting_station_index_args_raises():
    ds = _make_mock_dataset()
    with pytest.raises(ValueError):
        OATOOTGDataGenerator(
            dataset=ds,
            obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
            nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
            xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
            dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
            eligible_station_indices=np.array([0, 1, 2]),
            context_station_indices=np.array([0, 1]),
            target_station_indices=np.array([2, 3]),
        )


def test_ablate_obs_empties_context():
    ds = _make_mock_dataset(n_stations=20)
    ds.__getitem__.return_value["observation_data"]["mask_obs"][:] = 1.0
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        ablate_obs=True,
    )
    batch = gen.generate_batch()
    assert batch.xc.shape[1] == 0 and batch.yc.shape[1] == 0
    assert batch.xc.shape[2] == 2 and batch.yc.shape[2] == 1  # dims preserved
    assert batch.xt.shape[1] > 0  # targets untouched
    # NWP grid retained under obs ablation
    assert batch.yc_grid.abs().sum() > 0


def test_ablate_grid_zeros_nwp():
    ds = _make_mock_dataset(n_stations=20)
    ds.__getitem__.return_value["observation_data"]["mask_obs"][:] = 1.0
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        ablate_grid=True,
    )
    batch = gen.generate_batch()
    assert (batch.yc_grid == 0).all()  # NWP grid killed
    assert batch.xc.shape[1] > 0  # station context retained


def test_ablate_obs_forward_through_model():
    """Empty-context batch must propagate through the real OOTG model (nc=0)."""
    from oat.baselines.ootg_model import build_ootg_model

    ds = _make_mock_dataset(n_stations=20)
    ds.__getitem__.return_value["observation_data"]["mask_obs"][:] = 1.0
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        ablate_obs=True,
    )
    batch = gen.generate_batch()
    model = build_ootg_model(
        dim_y=1, dim_gridded_y=3,
        obs_variable_to_distribution={"temperature": "normal"},
        embed_dim=16, x_embed_dim=16, num_heads=2, head_dim=8, num_layers=1,
        num_legendre_polys=4, grid_shape=(8, 8), window_sizes=(4, 4),
        shift_sizes=(2, 2), top_k_ctot=4,
    )
    dist = model(batch.xc, batch.yc, batch.xc_grid, batch.yc_grid, batch.xt)
    assert dist.mean.shape[1] == batch.xt.shape[1]  # one prediction per target


def test_compute_norm_stats_shapes():
    ds = _make_mock_dataset()
    om, os_, nm, ns = compute_norm_stats(ds, n_samples=10)
    assert om.shape == (1,) and os_.shape == (1,)
    assert nm.shape == (3,) and ns.shape == (3,)
    assert os_.item() > 0 and (ns > 0).all()


def test_compute_norm_stats_multivariate():
    ds = _make_mock_dataset(n_obs=2)
    om, os_, nm, ns = compute_norm_stats(ds, n_samples=10)
    assert om.shape == (2,) and os_.shape == (2,)


def test_make_ootg_generator_cosmo():
    ds = _make_mock_dataset()
    obs_mean, obs_std = torch.tensor([280.0]), torch.tensor([5.0])
    nwp_mean, nwp_std = torch.zeros(3), torch.ones(3)
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


def test_topo_generator_returns_topo_batch():
    batch = _make_topo_generator().generate_batch()
    assert isinstance(batch, OOTGTopoBatch)


def test_topo_shapes_match_context_and_target():
    batch = _make_topo_generator().generate_batch()
    # topo_c/topo_t: [B=1, N, C_topo, Hp, Wp]; N matches xc/xt station counts
    assert batch.topo_c.shape[1] == batch.xc.shape[1]
    assert batch.topo_t.shape[1] == batch.xt.shape[1]
    assert batch.topo_c.shape[2:] == (C_TOPO, HP, WP)
    assert batch.topo_t.shape[2:] == (C_TOPO, HP, WP)


def test_no_topo_transform_returns_plain_batch():
    # topo_transform=None (default) -> unchanged behavior
    assert not isinstance(_make_generator().generate_batch(), OOTGTopoBatch)


def _sampled_time_indices(rank, n_batches=40, base_seed=0):
    """Time steps a generator draws when it believes it is DDP `rank`."""
    from unittest.mock import patch

    ds = _make_mock_dataset(n_stations=20)
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.tensor([280.0]), obs_std=torch.tensor([5.0]),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        time_indices=np.arange(50), seed=base_seed,
    )
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=rank),
    ):
        indices = []
        for _ in range(n_batches):
            gen.generate_batch()
            indices.append(int(ds.__getitem__.call_args[0][0]))
    return indices


def test_ddp_ranks_sample_different_timesteps():
    """Each DDP rank must draw different time steps, else all ranks compute
    identical batches and DDP averages duplicate (redundant) gradients."""
    rank0 = _sampled_time_indices(rank=0)
    rank1 = _sampled_time_indices(rank=1)
    assert rank0 != rank1, "ranks drew identical time-step sequences"
    frac_shared = sum(a == b for a, b in zip(rank0, rank1)) / len(rank0)
    assert frac_shared < 0.2, f"rank streams too correlated ({frac_shared:.2f})"


def test_per_rank_sampling_is_deterministic():
    """A given rank draws a reproducible sequence (needed for resumable runs)."""
    assert _sampled_time_indices(rank=0) == _sampled_time_indices(rank=0)


def test_ablate_obs_empty_topo_context():
    ds = _add_topo_to_dataset(_make_mock_dataset(20, n_obs=1), 20)
    gen = OATOOTGDataGenerator(
        dataset=ds,
        obs_mean=torch.full((1,), 280.0), obs_std=torch.full((1,), 5.0),
        nwp_mean=torch.zeros(3), nwp_std=torch.ones(3),
        xc_grid_latlon=_xc_grid(), coord_to_latlon=lv95_to_latlon,
        dist_names=["normal"], nwp_stride=2, samples_per_epoch=10,
        topo_transform=_topo_transform(), ablate_obs=True,
    )
    batch = gen.generate_batch()
    assert batch.topo_c.shape[1] == 0
    assert batch.topo_t.shape[1] == batch.xt.shape[1]
