import logging

import numpy as np
import torch
from omegaconf import OmegaConf

import scripts.evaluate_test_set as evaluate_test_set_module
from scripts.evaluate_test_set import (
    COSMO_INTERPOLATOR_MODEL,
    INTERPOLATOR_MODEL,
    _base_tag,
    _build_summary,
    _ckpt_tag,
    _compose_cosmo_interpolator_cfg,
    _cosmo_interpolator_dataset_is_valid,
    _exclude_context,
    _holdout_split_error,
    _missing_runs,
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


def test_cosmo_interpolator_dataset_is_valid_for_era5_meteostat():
    assert _cosmo_interpolator_dataset_is_valid("era5_meteostat") is True


def test_cosmo_interpolator_dataset_is_valid_rejects_other_datasets():
    assert _cosmo_interpolator_dataset_is_valid("cosmo_peakweather") is False
    assert _cosmo_interpolator_dataset_is_valid(None) is False


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


def test_holdout_split_error_requires_holdout_split_for_holdout_models():
    """A holdout model run without the holdout split writes a result file
    without the suffix and with another test window. The script must refuse
    this run."""
    for model in (
        "oat_spatial_holdout_temp_precip",
        "uninformed_spatial_holdout_temp_precip",
        COSMO_INTERPOLATOR_MODEL,
    ):
        assert _holdout_split_error(model, None) is not None
        assert _holdout_split_error(model, "other_split") is not None
        assert _holdout_split_error(model, "spatial_region_holdout") is None


def test_holdout_split_error_allows_other_models():
    assert _holdout_split_error("oat_temp_precip", None) is None
    assert _holdout_split_error(INTERPOLATOR_MODEL, None) is None
    assert _holdout_split_error(None, None) is None


def test_build_summary_includes_vs():
    metrics = {
        **_base_metrics(),
        "vs": torch.tensor([0.5]),
        "vs_count": torch.tensor([7.0]),
    }
    summary = _build_summary(metrics, ["prcp"])
    assert summary["vs"] == {"prcp": 0.5}
    assert summary["vs_count"] == {"prcp": 7}
    assert isinstance(summary["vs_count"]["prcp"], int)


def test_build_summary_without_vs_is_unchanged():
    summary = _build_summary(_base_metrics(), ["prcp"])
    assert "vs" not in summary
    assert "vs_count" not in summary
    assert set(summary) == {"crps", "mae", "crps_global", "mae_global"}


def _registry_fixture(tmp_path):
    """Make a small registry with two checkpoint files on disk."""
    ckpt_root = tmp_path / "ckpts"
    for rel in ("oat/run_a/best.ckpt", "oat/run_b/best.ckpt", "oat/run_h/best.ckpt"):
        (ckpt_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (ckpt_root / rel).touch()
    registry = {
        "cosmo_peakweather": {
            "interpolator": "interpolator",
            "model_a": "oat/run_a/best.ckpt",
            "model_b": "oat/run_b/best.ckpt",
            "not_set": None,
            "not_on_disk": "oat/run_x/best.ckpt",
        },
        "era5_meteostat": {
            "oat_spatial_holdout_temp_precip": "oat/run_h/best.ckpt",
            "cosmo_interpolator": "cosmo_interpolator",
        },
    }
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    return registry, ckpt_root, out_dir


def test_missing_runs_skips_evaluated_null_and_absent_checkpoints(tmp_path):
    registry, ckpt_root, out_dir = _registry_fixture(tmp_path)
    (out_dir / "run_a_metrics_test.csv").touch()
    runs = _missing_runs(
        registry, ["cosmo_peakweather"], "test", False, out_dir, ckpt_root
    )
    assert runs == [
        ("cosmo_peakweather", "interpolator", None),
        ("cosmo_peakweather", "model_b", None),
    ]


def test_missing_runs_sets_holdout_split_for_holdout_models(tmp_path):
    registry, ckpt_root, out_dir = _registry_fixture(tmp_path)
    (out_dir / "cosmo_interpolator_era5_meteostat_metrics_test.csv").touch()
    runs = _missing_runs(registry, None, "test", False, out_dir, ckpt_root)
    holdout = [r for r in runs if r[0] == "era5_meteostat"]
    assert holdout == [
        ("era5_meteostat", "oat_spatial_holdout_temp_precip", "spatial_region_holdout"),
        ("era5_meteostat", "cosmo_interpolator", "spatial_region_holdout"),
    ]
    (out_dir / "run_h_spatial_region_holdout_metrics_test.csv").touch()
    (
        out_dir / "cosmo_interpolator_era5_meteostat_spatial_region_holdout"
        "_metrics_test.csv"
    ).touch()
    assert not [
        r
        for r in _missing_runs(registry, None, "test", False, out_dir, ckpt_root)
        if r[0] == "era5_meteostat"
    ]


def test_missing_runs_uses_split_and_nocontext_in_file_name(tmp_path):
    registry, ckpt_root, out_dir = _registry_fixture(tmp_path)
    (out_dir / "run_a_metrics_test.csv").touch()
    (out_dir / "run_b_nocontext_metrics_val.csv").touch()
    runs = _missing_runs(
        registry, ["cosmo_peakweather"], "val", True, out_dir, ckpt_root
    )
    models = [r[1] for r in runs]
    assert "model_a" in models
    assert "model_b" not in models
