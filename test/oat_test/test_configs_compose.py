"""Every selectable dataset and model option composes with the root config."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
# Partial configs that other files include with "@_here_"; not selectable.
_NOT_SELECTABLE = {"base", "settings", "catalog", "plot_config"}


def _options(group: str) -> list[str]:
    root = CONFIG_DIR / group
    return sorted(
        str(p.relative_to(root).with_suffix(""))
        for p in root.rglob("*.yaml")
        if p.stem not in _NOT_SELECTABLE
    )


@pytest.mark.parametrize("option", _options("dataset"))
def test_dataset_option_composes(option):
    # If an option fails here and also fails on main, it is a separate bug.
    # Report it to the controller; do not delete or skip it silently.
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        compose(config_name="config", overrides=[f"dataset={option}"])


@pytest.mark.parametrize("option", _options("model/global_encoder"))
def test_global_encoder_option_composes(option):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=[f"model/global_encoder={option}"]
        )
    assert cfg.model.global_encoder.type == "Point"


def test_legacy_dataset_option_is_gone():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        with pytest.raises(Exception, match="legacy"):
            compose(
                config_name="config",
                overrides=["dataset=cosmo_peakweather/legacy/precip"],
            )
