import pytest
from hydra import compose, initialize_config_dir

from oat.paths import REPO_ROOT

_CONFIG_DIR = str(REPO_ROOT / "configs")
_PATH_KEYS = (
    "nwp_path",
    "topo_path",
    "obs_path",
    "station_meta_path",
    "nwp_extracted_path",
    "cache_dir",
)
_DATASETS = [
    "era5_meteostat/temp_precip",
    "era5_meteostat/temp_precip_ootg",
    "cosmo_peakweather/temp_precip",
    "cosmo_peakweather/temp_precip_ootg",
    "cosmo_meteostat/temp_precip",
]


def _compose(config_name, overrides):
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


@pytest.mark.parametrize("dataset", _DATASETS)
def test_dataset_paths_use_env_data_root(dataset, monkeypatch):
    monkeypatch.setenv("OAT_DATA_ROOT", "/data_root_test")
    cfg = _compose("config", [f"dataset={dataset}"])
    for key in _PATH_KEYS:
        if key in cfg.dataset:
            assert str(cfg.dataset[key]).startswith("/data_root_test/"), key


@pytest.mark.parametrize("dataset", _DATASETS)
def test_dataset_paths_default_to_relative_data(dataset, monkeypatch):
    monkeypatch.delenv("OAT_DATA_ROOT", raising=False)
    cfg = _compose("config", [f"dataset={dataset}"])
    for key in _PATH_KEYS:
        if key in cfg.dataset:
            assert str(cfg.dataset[key]).startswith("data/"), key


@pytest.mark.parametrize(
    "config_name", ["config", "uninformed_config", "ootg_baseline"]
)
def test_root_configs_have_paths_and_no_fixed_entity(config_name, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("OAT_CKPT_ROOT", "/ckpt_root_test")
    cfg = _compose(config_name, [])
    assert cfg.paths.checkpoint_root == "/ckpt_root_test"
    assert cfg.logging.entity is None


def _patch_api_key(monkeypatch, key):
    import wandb

    monkeypatch.setattr(type(wandb.api), "api_key", property(lambda self: key))


def test_logger_disabled_without_login(monkeypatch, tmp_path):
    from scripts.train_model import setup_logger_and_callbacks

    _patch_api_key(monkeypatch, None)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    cfg = _compose(
        "config",
        [f"paths.checkpoint_root={tmp_path}", "paths.run_root=x", "run_name=t"],
    )
    logger, _ = setup_logger_and_callbacks(cfg)
    assert logger._wandb_init["mode"] == "disabled"
    assert (tmp_path / cfg.logging.group / "t").is_dir()


def test_logger_keeps_user_wandb_mode(monkeypatch, tmp_path):
    from scripts.train_model import setup_logger_and_callbacks

    _patch_api_key(monkeypatch, None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    cfg = _compose(
        "config",
        [f"paths.checkpoint_root={tmp_path}", "paths.run_root=x", "run_name=t"],
    )
    logger, _ = setup_logger_and_callbacks(cfg)
    assert logger._wandb_init.get("mode") is None
