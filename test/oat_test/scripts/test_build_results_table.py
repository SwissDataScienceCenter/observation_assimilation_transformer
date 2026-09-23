from scripts.build_results_table import _resolve_tag


def test_resolve_tag_sentinel_without_split_config():
    assert _resolve_tag("era5_meteostat", "interpolator", "interpolator") == (
        "interpolator_era5_meteostat"
    )


def test_resolve_tag_interpolator_appends_split_config_suffix():
    """A --split-config-aware sentinel must resolve to the same suffixed
    tag evaluate_test_set.py's _base_tag produced for that run.

    Without this, build_results_table.py silently falls back to a stale,
    unsuffixed result file from an earlier run on a different split.
    """
    assert _resolve_tag(
        "era5_meteostat", "interpolator", "interpolator",
        "spatial_region_holdout",
    ) == "interpolator_era5_meteostat_spatial_region_holdout"


def test_resolve_tag_cosmo_interpolator_ignores_split_config():
    """cosmo_interpolator hardcodes its own split internally (it composes
    a fresh Hydra config with split=spatial_region_holdout baked in, not
    via --split-config), so its result file is never suffixed -- unlike
    plain interpolator, which needs --split-config to reach that split."""
    assert _resolve_tag(
        "era5_meteostat", "cosmo_interpolator", "cosmo_interpolator",
        "spatial_region_holdout",
    ) == "cosmo_interpolator_era5_meteostat"


def test_resolve_tag_checkpoint_path_ignores_split_config():
    """A real checkpoint path's tag is its own directory name, regardless
    of split_config -- only sentinel entries are suffixed."""
    value = "/checkpoints/oat_field/oat_era5_meteostat-temp_precip_run/best.ckpt"
    assert (
        _resolve_tag("era5_meteostat", "oat_temp_precip", value, "spatial_region_holdout")
        == "oat_era5_meteostat-temp_precip_run"
    )


def test_resolve_tag_none_value_returns_none():
    assert _resolve_tag("era5_meteostat", "oat_temp_precip", None) is None
