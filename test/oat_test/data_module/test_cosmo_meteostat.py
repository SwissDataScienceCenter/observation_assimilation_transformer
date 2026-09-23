import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from oat.data_module.field_spec import parse_field_config, validate_field_config
from oat.data_module.sources import SourceInfo

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CONFIG_DIR = os.path.join(_REPO_ROOT, "configs")

_settings_path = os.path.join(_REPO_ROOT, "configs/dataset/era5_meteostat/base.yaml")
_TOPO_PATH = str(OmegaConf.load(_settings_path).topo_path)
_NWP_PATH = "/store_new/mch/msopr/ml/datasets/mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"


def _compose_dataset():
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(
            config_name="config", overrides=["dataset=cosmo_meteostat/temp_precip"]
        ).dataset


def test_temp_precip_config_parses_and_validates():
    cfg = _compose_dataset()
    assert cfg.dataset_type == "cosmo_meteostat"
    assert list(cfg.obs_vars) == ["temp", "prcp"]
    fc = parse_field_config(cfg)
    assert fc.roles["target"] == "obs"
    cat = cfg.catalog
    sources = {
        "obs": SourceInfo(list(cat.obs_vars.keys()), "stations", True, {}),
        "nwp": SourceInfo(list(cat.nwp_vars.keys()), "nwp_grid", False, {}),
        "topo": SourceInfo(list(cat.topo_vars.keys()), None, False, {}),
    }
    validate_field_config(fc, sources)  # no raise


def test_composed_with_spatial_holdout_split_pulls_in_meteostat_stations():
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cosmo_meteostat/temp_precip",
                "split=spatial_region_holdout",
            ],
        )
    # a known station ID from configs/split/spatial_region_holdout.yaml
    assert "02926" in list(cfg.split.train_station_indices)


@pytest.mark.skipif(not os.path.exists(_TOPO_PATH), reason="Meteostat topo data not present")
@pytest.mark.skipif(not os.path.exists(_NWP_PATH), reason="COSMO-E1 NWP archive not present")
def test_cosmo_meteostat_field_dataset_smoke():
    from oat.data_module.cosmo_meteostat_dataset import CosmoMeteostatDataset
    from oat.data_module.field import Field

    ds = CosmoMeteostatDataset(_compose_dataset())
    sample = ds[0]
    fields = sample["fields"]
    assert isinstance(fields["obs"], Field)
    assert fields["obs"].coords.shape[-1] == 3
    assert fields["obs"].mask is not None
