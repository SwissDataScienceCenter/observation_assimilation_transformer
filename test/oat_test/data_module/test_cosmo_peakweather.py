import os

import pytest
import torch
from omegaconf import OmegaConf

from oat.data_module.field import Field

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATA_DIR = "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked"
_DATA = os.path.join(_DATA_DIR, "nwp_extracted.zarr")
pytestmark = pytest.mark.skipif(
    not os.path.exists(_DATA), reason="COSMO data not present"
)

_FIELD_SCHEMA = {
    "obs_vars": ["temperature"],
    "nwp_vars": ["T_2M"],
    "coord_sources": {
        "stations": {"from": "obs"},
        "nwp_grid": {"from": "nwp"},
    },
    "fields": {
        "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
        "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
        "topo": {"coords": "stations", "inputs": ["topo.DEM"], "query": True},
        "geo": {
            "coords": "stations",
            "inputs": ["coords.x", "coords.y", "topo.DEM"],
            "out": ["swiss_easting", "swiss_northing", "dem"],
            "query": True,
        },
    },
    "roles": {"target": "obs", "geo_bias": "geo", "topo": "topo"},
}


def _config():
    base = OmegaConf.load(
        os.path.join(_REPO_ROOT, "configs/dataset/cosmo_peakweather/base.yaml")
    )
    settings = OmegaConf.load(
        os.path.join(_REPO_ROOT, "configs/dataset/cosmo_peakweather/settings.yaml")
    )
    catalog = OmegaConf.load(
        os.path.join(_REPO_ROOT, "configs/dataset/cosmo_peakweather/catalog.yaml")
    )
    return OmegaConf.merge(
        base,
        settings,
        OmegaConf.create({"catalog": catalog}),
        OmegaConf.create(_FIELD_SCHEMA),
    )


def test_cosmo_field_dataset_smoke():
    from oat.data_module.cosmo_peakweather_dataset import (
        CosmoPeakWeatherDataset,
    )

    ds = CosmoPeakWeatherDataset(_config())
    sample = ds[0]
    fields = sample["fields"]

    assert isinstance(fields["obs"], Field)
    assert fields["obs"].coords.shape[-1] == 3
    assert fields["obs"].mask is not None

    query_coords = torch.as_tensor(ds._coords["stations"][:3])
    q = ds.get_query(query_coords, idx=0)
    assert set(q) <= set(fields)
    assert "obs" not in q
