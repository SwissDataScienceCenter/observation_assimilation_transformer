import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf


def _make_cfg(tmp_path):
    # Synthetic obs parquet
    pd.DataFrame({
        "station": ["S1", "S2"], "time": ["2020-01-01 00:00:00", "2020-01-01 00:00:00"],
        "temp": [5.0, 6.0], "rhum": [80.0, 75.0], "prcp": [0.0, 0.1],
        "wspd": [3.0, 4.0], "wpgt": [float("nan"), 8.0],
    }).to_parquet(tmp_path / "obs.parquet", index=False)
    pd.DataFrame({
        "station": ["S1", "S2"],
        "latitude": [47.0, 48.0], "longitude": [8.0, 9.0], "elevation": [500.0, 600.0],
        "tier": ["wmo", "wmo"], "country": ["CH", "DE"],
    }).to_parquet(tmp_path / "meta.parquet", index=False)

    # Synthetic topo zarr (lat descending as rasterio produces)
    topo_lat = np.linspace(72.0, 34.0, 20)  # descending N→S
    topo_lon = np.linspace(-25.0, 50.0, 30)
    topo_ds = xr.Dataset(
        {
            "elevation": (["latitude", "longitude"],
                          np.random.rand(20, 30).astype(np.float32) * 2000),
            "distance_to_coast": (["latitude", "longitude"],
                                  np.random.rand(20, 30).astype(np.float32) * 500_000),
        },
        coords={"latitude": topo_lat, "longitude": topo_lon},
    )
    topo_ds.to_zarr(tmp_path / "topo.zarr", mode="w")

    return OmegaConf.create({
        "nwp_path": "fake_era5.zarr",
        "topo_path": str(tmp_path / "topo.zarr"),
        "obs_path": str(tmp_path / "obs.parquet"),
        "station_meta_path": str(tmp_path / "meta.parquet"),
        "nwp_vars": ["2t", "10u"],
        "topo_vars": ["elevation"],
        "obs_vars": ["temp", "wspd"],
        "coord_vars": ["longitude", "latitude"],
        "geo_vars": ["longitude", "latitude", "elevation"],
        "stations": None,
        "start_date": "2020-01-01",
        "end_date": "2020-01-01",
        "nwp_lat_min": 34.0, "nwp_lat_max": 72.0,
        "nwp_lon_min": -25.0, "nwp_lon_max": 50.0,
        "nwp_lat_res": 5.0, "nwp_lon_res": 5.0,
        "patch_size": [8, 8],
        "use_time_embedding": True,
        "dtype": "float32",
        "data_points_for_stats": 2,
        "catalog": OmegaConf.create({
            "obs_vars": {
                "temp": {"lower_bound": float("-inf"), "upper_bound": float("inf")},
                "wspd": {"lower_bound": 0.0, "upper_bound": float("inf"), "lower_open": True},
            }
        }),
    })


@pytest.fixture
def mock_nwp():
    n = 30
    ds = MagicMock()
    ds.name_to_index = {"2t": 0, "10u": 1}
    ds.latitudes = np.linspace(34.0, 72.0, n)
    ds.longitudes = np.linspace(-25.0, 50.0, n)
    ds.dates = np.array(["2020-01-01T00:00:00"], dtype="datetime64[s]")
    ds.missing = None
    ds.__getitem__ = MagicMock(return_value=np.random.rand(2, 1, n).astype(np.float32))
    return ds


@pytest.fixture
def dataset(tmp_path, mock_nwp):
    from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset
    with patch("oat.data_module.era5_meteostat_dataset.open_dataset", return_value=mock_nwp):
        return ERA5MeteostatDataset(_make_cfg(tmp_path))


def test_len(dataset):
    assert len(dataset) >= 1


def test_getitem_keys(dataset):
    sample = dataset[0]
    assert set(sample) >= {"index", "global_data", "observation_data"}
    assert "global_field" in sample["global_data"]
    assert "time_emb" in sample["global_data"]
    for k in ("coords", "geo", "topo", "obs", "mask_obs"):
        assert k in sample["observation_data"]


def test_global_field_channels(dataset):
    gf = dataset[0]["global_data"]["global_field"]
    assert gf.ndim == 3
    assert gf.shape[0] == len(dataset.config.nwp_vars) + len(dataset.config.topo_vars)
    assert gf.dtype == torch.float32


def test_observation_shapes(dataset):
    od = dataset[0]["observation_data"]
    N = len(dataset.stations)
    assert od["coords"].shape == (N, 2)
    assert od["geo"].shape == (N, 3)
    assert od["obs"].shape == (N, 2)
    assert od["mask_obs"].shape == (N, 2)
    assert od["topo"].shape == (N, 1, 8, 8)


def test_nwp_extent(dataset):
    assert "min" in dataset.nwp_extent and "max" in dataset.nwp_extent
    assert dataset.nwp_extent["min"].shape == (2,)


def test_stations_is_array(dataset):
    assert isinstance(dataset.stations, np.ndarray)
    assert len(dataset.stations) == 2
