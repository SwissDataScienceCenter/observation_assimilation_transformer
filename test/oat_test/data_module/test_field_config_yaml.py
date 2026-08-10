import os

from hydra import compose, initialize_config_dir

from oat.data_module.field_spec import parse_field_config, validate_field_config
from oat.data_module.sources import SourceInfo


def _load(name):
    d = os.path.abspath("configs/dataset/cosmo_peakweather")
    with initialize_config_dir(version_base=None, config_dir=d):
        return compose(config_name=name)


def test_field_temp_config_parses_and_validates():
    cfg = _load("field_temp")
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
