"""The train dispatcher wires the flat-Field datasets + point-wise encoder models."""

import os

import pytest
from hydra import compose, initialize_config_dir

from scripts.train import (
    DATASETS,
    MODEL_SPECS,
    build_overrides,
    validate,
)


def test_datasets_are_selectable():
    assert "cosmo_peakweather/temp_precip" in DATASETS
    assert "cosmo_peakweather/temp" in DATASETS
    assert "era5_meteostat/temp_precip" in DATASETS


def test_oat_field_bundles_point_and_mlp_encoders():
    assert "oat_field" in MODEL_SPECS
    assert "uninformed_oat_field" in MODEL_SPECS
    ov = build_overrides("oat_field", "cosmo_peakweather/temp_precip", "run", extra=[])
    assert "model/global_encoder=point" in ov
    assert "model/topographic_encoder=mlp" in ov
    assert "dataset=cosmo_peakweather/temp_precip" in ov


def test_field_and_non_field_models_accept_the_same_dataset():
    # Every non-legacy dataset is flat-Field-shaped now, so there is no
    # dataset-naming split left to gate on: both the "_field" preset and the
    # plain model accept the same dataset variant without raising.
    validate("oat_field", "cosmo_peakweather/temp_precip")
    validate("oat", "cosmo_peakweather/temp_precip")


def test_dispatcher_overrides_compose_into_field_config():
    ov = build_overrides("oat_field", "cosmo_peakweather/temp_precip", "run", extra=[])
    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="config", overrides=ov)
    assert cfg.dataset.dataset_type == "cosmo_peakweather"
    assert cfg.model.global_encoder.type == "Point"
    assert cfg.model.topographic_encoder.type == "MLP"
    assert cfg.model.transform_type == "keep_support"
    assert dict(cfg.model.output_head.variable_to_distribution) == dict(
        cfg.dataset.obs_variable_to_distribution
    )


def test_gp_field_variants_present():
    for name in ("gp_field", "deep_gp_field", "gp_oat_field", "deep_gp_oat_field"):
        assert name in MODEL_SPECS
        assert MODEL_SPECS[name].univariate_only
    for old_name in ("gp", "deep_gp", "gp_oat", "deep_gp_oat"):
        assert old_name not in MODEL_SPECS


def test_gp_field_bundles_mlp_topo_encoder():
    ov = build_overrides("gp_field", "cosmo_peakweather/temp", "run", extra=[])
    assert "model/topographic_encoder=mlp" in ov
    assert "trainer.precision=64" in ov
    assert "dataset=cosmo_peakweather/temp" in ov


def test_gp_oat_field_bundles_point_and_mlp_encoders():
    ov = build_overrides("gp_oat_field", "cosmo_peakweather/temp", "run", extra=[])
    assert "model/global_encoder=point" in ov
    assert "model/topographic_encoder=mlp" in ov


def test_gp_field_model_requires_univariate_dataset():
    with pytest.raises(SystemExit):
        validate("gp_field", "cosmo_peakweather/temp_precip")
    validate("gp_field", "cosmo_peakweather/temp")


def test_ootg_datasets_are_selectable():
    assert "cosmo_peakweather/temp_ootg" in DATASETS
    assert "cosmo_peakweather/temp_precip_ootg" in DATASETS
    assert "era5_meteostat/temp_ootg" in DATASETS
    assert "era5_meteostat/temp_precip_ootg" in DATASETS


def test_ootg_model_requires_ootg_dataset_variant():
    with pytest.raises(SystemExit):
        validate("ootg", "cosmo_peakweather/temp")
    with pytest.raises(SystemExit):
        validate("ootg", "cosmo_peakweather/temp_precip")
    validate("ootg", "cosmo_peakweather/temp_ootg")
    validate("ootg", "cosmo_peakweather/temp_precip_ootg")
