import json
import math

import pytest

import scripts.build_results_table as build_results_table_module
from scripts.build_results_table import _collect, _resolve_tag, _tag_for_run


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


def _write_summary(results_dir, tag, crps, split="test"):
    path = results_dir / f"{tag}_metrics_{split}.summary.json"
    path.write_text(json.dumps({"crps": {"temp": crps}, "mae": {"temp": crps}}))


def _holdout_row(data, base):
    rows = data[build_results_table_module.SPATIAL_HOLDOUT_VIEW]
    return next(r for r in rows if r["base"] == base)


def test_collect_holdout_view_reads_only_suffixed_files(tmp_path, monkeypatch):
    """The holdout view must read the _spatial_region_holdout file of every
    model. A file without the suffix has a different test window, so it must
    not be used, also not as a fallback: a missing file gives NA."""
    monkeypatch.setattr(build_results_table_module, "RESULTS_DIR", tmp_path)
    registry = {
        "era5_meteostat": {
            "interpolator": "interpolator",
            "cosmo_interpolator": "cosmo_interpolator",
            "oat_spatial_holdout_temp_precip": "/c/oat_holdout_run/best.ckpt",
            "uninformed_spatial_holdout_temp_precip": "/c/uninf_holdout_run/b.ckpt",
        }
    }
    suffix = "_spatial_region_holdout"
    _write_summary(tmp_path, "oat_holdout_run" + suffix, 0.8)
    _write_summary(tmp_path, "oat_holdout_run", 9.9)
    _write_summary(tmp_path, "cosmo_interpolator_era5_meteostat" + suffix, 1.2)
    _write_summary(tmp_path, "cosmo_interpolator_era5_meteostat", 9.9)
    _write_summary(tmp_path, "uninf_holdout_run", 9.9)  # unsuffixed only

    data = _collect(registry, "test", 10, 0, (2.5, 97.5))

    assert _holdout_row(data, "oat_spatial_holdout")["vars"]["temperature"][
        "crps"
    ] == pytest.approx(0.8)
    assert _holdout_row(data, "cosmo_interpolator")["vars"]["temperature"][
        "crps"
    ] == pytest.approx(1.2)
    assert math.isnan(
        _holdout_row(data, "uninformed_spatial_holdout")["vars"]["temperature"][
            "crps"
        ]
    )
