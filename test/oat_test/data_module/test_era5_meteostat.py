import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from oat.data_module.field import Field
from oat.data_module.field_spec import parse_field_config, validate_field_config
from oat.data_module.sources import SourceInfo

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

_settings_path = os.path.join(_REPO_ROOT, "configs/dataset/era5_meteostat/base.yaml")
_TOPO_PATH = str(OmegaConf.load(_settings_path).topo_path)
pytestmark = pytest.mark.skipif(
    not os.path.exists(_TOPO_PATH), reason="ERA5 topo data not present"
)


def _compose_dataset():
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(
            config_name="config", overrides=["dataset=era5_meteostat/temp"]
        ).dataset


def test_temp_config_parses_and_validates():
    cfg = _compose_dataset()
    fc = parse_field_config(cfg)
    assert fc.transform_type == "real"
    assert fc.roles["target"] == "obs"
    # validate against the catalog-declared vars
    cat = cfg.catalog
    sources = {
        "obs": SourceInfo(list(cat.obs_vars.keys()), "stations", True, {}),
        "nwp": SourceInfo(list(cat.nwp_vars.keys()), "nwp_grid", False, {}),
        "topo": SourceInfo(list(cat.topo_vars.keys()), None, False, {}),
    }
    validate_field_config(fc, sources)  # no raise


_FIELD_SCHEMA = {
    "obs_vars": ["temp"],
    "nwp_vars": ["2t"],
    "coord_sources": {
        "stations": {"from": "obs"},
        "nwp_grid": {"from": "nwp"},
    },
    "fields": {
        "obs": {"coords": "stations", "inputs": ["obs.temp"]},
        "nwp": {"coords": "stations", "inputs": ["nwp.2t"], "query": True},
        "topo": {"coords": "stations", "inputs": ["topo.elevation"], "query": True},
        "geo": {
            "coords": "stations",
            "inputs": ["coords.x", "coords.y", "topo.elevation"],
            "out": ["longitude", "latitude", "elevation"],
            "query": True,
        },
    },
    "roles": {"target": "obs", "geo_bias": "geo", "topo": "topo"},
    "transform_type": "real",
}


def _smoke_config():
    settings = OmegaConf.load(_settings_path)
    catalog = OmegaConf.load(
        os.path.join(_REPO_ROOT, "configs/dataset/era5_meteostat/catalog.yaml")
    )
    return OmegaConf.merge(
        settings,
        OmegaConf.create({"catalog": catalog}),
        OmegaConf.create(_FIELD_SCHEMA),
    )


def test_era5_field_dataset_smoke():
    from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset

    ds = ERA5MeteostatDataset(_smoke_config())
    sample = ds[0]
    fields = sample["fields"]

    assert isinstance(fields["obs"], Field)
    assert fields["obs"].coords.shape[-1] == 3
    assert fields["obs"].mask is not None

    query_coords = torch.as_tensor(ds._coords["stations"][:3])
    q = ds.get_query(query_coords, idx=0)
    assert set(q) <= set(fields)
    assert "obs" not in q
