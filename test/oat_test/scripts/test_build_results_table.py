import scripts.build_results_table as build_results_table_module
from scripts.build_results_table import _resolve_tag, _tag_for_run


def test_resolve_tag_sentinel():
    assert _resolve_tag("era5_meteostat", "interpolator", "interpolator") == (
        "interpolator_era5_meteostat"
    )


def test_resolve_tag_checkpoint_path():
    value = "/checkpoints/oat_field/oat_era5_meteostat-temp_precip_run/best.ckpt"
    assert (
        _resolve_tag("era5_meteostat", "oat_temp_precip", value)
        == "oat_era5_meteostat-temp_precip_run"
    )


def test_resolve_tag_none_value_returns_none():
    assert _resolve_tag("era5_meteostat", "oat_temp_precip", None) is None


def _touch_summary(results_dir, tag, split="test"):
    (results_dir / f"{tag}_metrics_{split}.summary.json").write_text("{}")


def test_tag_for_run_prefers_suffixed_file_when_it_exists(tmp_path, monkeypatch):
    """A registry entry actually evaluated with --split-config must resolve
    to that suffixed file, not the plain one -- this is what keeps a table
    from silently reading a stale, differently-windowed result."""
    monkeypatch.setattr(build_results_table_module, "RESULTS_DIR", tmp_path)
    _touch_summary(tmp_path, "interpolator_era5_meteostat")
    _touch_summary(tmp_path, "interpolator_era5_meteostat_spatial_region_holdout")
    tag = _tag_for_run(
        "era5_meteostat",
        "interpolator",
        "interpolator",
        "test",
        "spatial_region_holdout",
    )
    assert tag == "interpolator_era5_meteostat_spatial_region_holdout"


def test_tag_for_run_falls_back_to_unsuffixed_when_split_config_does_not_apply(
    tmp_path, monkeypatch, capsys
):
    """A --split-config that doesn't apply to this entry (e.g. a split
    that only exists for a different dataset) must fall back to the
    entry's normal result file with a warning, not silently blank it to
    NA."""
    monkeypatch.setattr(build_results_table_module, "RESULTS_DIR", tmp_path)
    _touch_summary(tmp_path, "interpolator_cosmo_peakweather")
    tag = _tag_for_run(
        "cosmo_peakweather",
        "interpolator",
        "interpolator",
        "test",
        "spatial_region_holdout",
    )
    assert tag == "interpolator_cosmo_peakweather"
    assert "falling back" in capsys.readouterr().err


def test_tag_for_run_no_split_config_returns_base_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(build_results_table_module, "RESULTS_DIR", tmp_path)
    tag = _tag_for_run("era5_meteostat", "interpolator", "interpolator", "test", None)
    assert tag == "interpolator_era5_meteostat"


def test_tag_for_run_checkpoint_path_also_prefers_suffixed_file(tmp_path, monkeypatch):
    """A real checkpoint re-evaluated with --split-config must be found
    too -- the suffix isn't sentinel-only."""
    monkeypatch.setattr(build_results_table_module, "RESULTS_DIR", tmp_path)
    value = "/checkpoints/oat_field/oat_era5_meteostat-temp_precip_run/best.ckpt"
    _touch_summary(
        tmp_path, "oat_era5_meteostat-temp_precip_run_spatial_region_holdout"
    )
    tag = _tag_for_run(
        "era5_meteostat",
        "oat_temp_precip",
        value,
        "test",
        "spatial_region_holdout",
    )
    assert tag == "oat_era5_meteostat-temp_precip_run_spatial_region_holdout"
