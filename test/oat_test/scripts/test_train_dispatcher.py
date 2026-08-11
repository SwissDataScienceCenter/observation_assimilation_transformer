"""The train dispatcher wires the field datasets + point-wise encoder models."""

import os

import pytest
from hydra import compose, initialize_config_dir

from scripts.train import (
    DATASETS,
    MODEL_SPECS,
    build_overrides,
    validate,
)


def test_field_datasets_are_selectable():
    assert "cosmo_peakweather/field_temp_precip" in DATASETS
    assert "cosmo_peakweather/field_temp" in DATASETS
    assert "era5_meteostat/field_temp_precip" in DATASETS


def test_oat_field_bundles_point_and_mlp_encoders():
    assert "oat_field" in MODEL_SPECS
    assert "uninformed_oat_field" in MODEL_SPECS
    ov = build_overrides(
        "oat_field", "cosmo_peakweather/field_temp_precip", "run", extra=[]
    )
    assert "model/global_encoder=point" in ov
    assert "model/topographic_encoder=mlp" in ov
    assert "dataset=cosmo_peakweather/field_temp_precip" in ov


def test_field_model_requires_field_dataset():
    with pytest.raises(SystemExit):
        validate("oat_field", "cosmo_peakweather/temp_precip")
    with pytest.raises(SystemExit):
        validate("oat", "cosmo_peakweather/field_temp_precip")
    # Correct pairings do not raise.
    validate("oat_field", "cosmo_peakweather/field_temp_precip")
    validate("oat", "cosmo_peakweather/temp_precip")


def test_dispatcher_overrides_compose_into_field_config():
    ov = build_overrides(
        "oat_field", "cosmo_peakweather/field_temp_precip", "run", extra=[]
    )
    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=ov)
    assert cfg.dataset.dataset_type == "field_cosmo_peakweather"
    assert cfg.model.global_encoder.type == "Point"
    assert cfg.model.topographic_encoder.type == "MLP"
    assert cfg.dataset.transform_type == "keep_support"
    assert dict(cfg.model.output_head.variable_to_distribution) == dict(
        cfg.dataset.obs_variable_to_distribution
    )
