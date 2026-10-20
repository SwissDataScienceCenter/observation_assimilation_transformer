from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

FULL = "${paths.data_root}/cosmo-1e_full_domain/"
ARCHIVE = "mch-co1e-an-archive-0p01-2019-2024-1h-v1-pl13.zarr"


def _compose(override: str) -> dict:
    """Return the dataset config with its interpolations left as text."""
    config_dir = str(Path(__file__).parents[2] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[override])
    ds = OmegaConf.to_container(cfg.dataset, resolve=False)
    assert isinstance(ds, dict)
    return ds


@pytest.mark.parametrize("variant", ["temp_ootg", "temp_precip_ootg"])
def test_ootg_configs_use_the_full_domain(variant):
    ds = _compose(f"dataset=cosmo_peakweather/{variant}")
    assert ds["nwp_path"] == FULL + ARCHIVE
    assert ds["nwp_extracted_path"] == FULL + "nwp_extracted.zarr"


@pytest.mark.parametrize("variant", ["temp", "precip", "temp_precip"])
def test_default_swiss_configs_use_the_default_names(variant):
    ds = _compose(f"dataset=cosmo_peakweather/{variant}")
    assert "cosmo-1e_full_domain" not in ds["nwp_path"] + ds["nwp_extracted_path"]


def test_cosmo_meteostat_uses_the_full_archive():
    ds = _compose("dataset=cosmo_meteostat/temp_precip")
    assert ds["nwp_path"] == FULL + ARCHIVE
