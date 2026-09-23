import logging

import numpy as np
import torch
from omegaconf import OmegaConf

import scripts.evaluate_test_set as evaluate_test_set_module
from scripts.evaluate_test_set import (
    COSMO_INTERPOLATOR_MODEL,
    INTERPOLATOR_MODEL,
    _base_tag,
    _ckpt_tag,
    _compose_cosmo_interpolator_cfg,
    _exclude_context,
    _override_split,
    _print_summary,
    _sibling_checkpoint,
)


def test_exclude_context_removes_context_stations():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({"test_context_station_indices": ["b", "d"]})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 2]


def test_exclude_context_noop_without_context_field():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 1, 2, 3]


def test_ckpt_tag_differs_with_and_without_exclude_context():
    """The output filename must differ between the two eval arms.

    Without this, a run with --exclude-context overwrites the CSV and
    summary from a run without it (and vice versa).
    """
    tag_with_context = _ckpt_tag("my_ckpt", exclude_context=False)
    tag_without_context = _ckpt_tag("my_ckpt", exclude_context=True)
    assert tag_with_context != tag_without_context
    assert tag_with_context == "my_ckpt"
    assert tag_without_context == "my_ckpt_nocontext"


def _base_metrics() -> dict:
    return {
        "crps": torch.zeros(2, 1),
        "mae": torch.zeros(2, 1),
        "crps_global": torch.tensor(0.0),
        "mae_global": torch.tensor(0.0),
    }


def test_print_summary_warns_on_nonfinite(caplog):
    """A non-finite prediction must surface as a loud warning, not silently
    vanish into the averaged CRPS/MAE numbers."""
    metrics = {**_base_metrics(), "n_nonfinite": torch.tensor([[1.0], [0.0]])}
    logger = logging.getLogger("test_print_summary_warns")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(metrics, ["prcp"], logger)
    assert "non-finite" in caplog.text


def test_print_summary_silent_when_all_finite(caplog):
    metrics = {**_base_metrics(), "n_nonfinite": torch.zeros(2, 1)}
    logger = logging.getLogger("test_print_summary_silent")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(metrics, ["prcp"], logger)
    assert "non-finite" not in caplog.text


def test_print_summary_handles_missing_n_nonfinite(caplog):
    """Callers that skip compute_station_metrics (the long-dump aggregation
    path) don't produce n_nonfinite at all; must not crash or warn."""
    logger = logging.getLogger("test_print_summary_missing")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(_base_metrics(), ["prcp"], logger)
    assert "non-finite" not in caplog.text


def test_base_tag_for_interpolator():
    assert (
        _base_tag(True, INTERPOLATOR_MODEL, "era5_meteostat", None)
        == "interpolator_era5_meteostat"
    )


def test_base_tag_for_cosmo_interpolator():
    assert (
        _base_tag(True, COSMO_INTERPOLATOR_MODEL, "era5_meteostat", None)
        == "cosmo_interpolator_era5_meteostat"
    )


def test_base_tag_for_checkpoint():
    checkpoint = "/checkpoints/oat_field/oat_era5_meteostat_run5379412/best.ckpt"
    assert _base_tag(False, "oat", None, checkpoint) == "oat_era5_meteostat_run5379412"


def test_compose_cosmo_interpolator_cfg_uses_cosmo_dataset_and_spatial_split():
    cfg = _compose_cosmo_interpolator_cfg()
    assert cfg.dataset.dataset_type == "cosmo_meteostat"
    assert "02926" in list(cfg.split.train_station_indices)
    assert list(cfg.dataset.obs_vars) == ["temp", "prcp"]


def test_base_tag_appends_split_config_when_given():
    assert (
        _base_tag(
            True,
            INTERPOLATOR_MODEL,
            "era5_meteostat",
            None,
            "spatial_region_holdout",
        )
        == "interpolator_era5_meteostat_spatial_region_holdout"
    )


def test_base_tag_omits_split_config_suffix_when_none():
    assert (
        _base_tag(True, INTERPOLATOR_MODEL, "era5_meteostat", None, None)
        == "interpolator_era5_meteostat"
    )


def test_override_split_replaces_cfg_split_with_named_config():
    """--split-config must pin the split regardless of what cfg already had.

    Uses a real Hydra compose of configs/split/spatial_region_holdout.yaml
    (no mocks) and checks for a known station ID from that file.
    """
    cfg = OmegaConf.create({"split": {"train_station_indices": None}})
    cfg = _override_split(cfg, "spatial_region_holdout")
    assert "02926" in list(cfg.split.train_station_indices)


def test_sibling_checkpoint_skips_both_interpolator_sentinels(tmp_path, monkeypatch):
    """A registry with two sentinel entries must not return a sentinel value.

    Regression test: adding cosmo_interpolator right after interpolator in
    checkpoints.yaml made _sibling_checkpoint return the literal string
    "cosmo_interpolator" instead of skipping past it to the real checkpoint
    path, breaking the pre-existing --model interpolator invocation.
    """
    registry_path = tmp_path / "checkpoints.yaml"
    registry_path.write_text(
        "era5_meteostat:\n"
        "  interpolator: interpolator\n"
        "  cosmo_interpolator: cosmo_interpolator\n"
        "  oat_temp_precip: /checkpoints/oat_field/oat_era5_meteostat_run1/best.ckpt\n"
    )
    monkeypatch.setattr(evaluate_test_set_module, "_DEFAULT_REGISTRY", registry_path)
    result = _sibling_checkpoint("era5_meteostat")
    assert result == "/checkpoints/oat_field/oat_era5_meteostat_run1/best.ckpt"
