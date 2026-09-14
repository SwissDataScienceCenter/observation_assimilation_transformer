"""Tests for scripts/generate_region_holdout_split.py's pure split logic.

No real data files needed: uses a small synthetic pandas DataFrame in place
of the real station_quality.parquet-derived lat/lon lookup.
"""

import pandas as pd
import pytest
import yaml

from scripts.generate_region_holdout_split import (
    compute_region_holdout_split,
    render_config,
)


def _coords(**rows):
    return pd.DataFrame.from_dict(rows, orient="index", columns=["latitude", "longitude"])


def test_holds_out_every_station_in_the_box():
    coords = _coords(
        in1=(45.0, 7.0), in2=(46.0, 8.0), in3=(44.5, 9.0),
        out1=(50.0, 7.0), out2=(52.0, 10.0), out3=(48.5, 20.0), out4=(60.0, 5.0),
    )
    train_ids, val_ids, ctx_ids, eval_ids = compute_region_holdout_split(
        coords, lat_min=44, lat_max=48, lon_min=5, lon_max=10.5,
        context_fraction=2 / 3, val_station_fraction=0.25, seed=0,
    )
    assert set(ctx_ids) | set(eval_ids) == {"in1", "in2", "in3"}
    assert set(train_ids) | set(val_ids) == {"out1", "out2", "out3", "out4"}
    assert set(ctx_ids).isdisjoint(eval_ids)
    assert set(train_ids).isdisjoint(val_ids)
    assert len(eval_ids) == 1  # int(3 * (1 - 2/3)) == 1


def test_deterministic_for_fixed_seed():
    coords = _coords(**{f"out{i}": (50.0, 7.0) for i in range(20)}, in1=(45.0, 7.0), in2=(45.5, 7.5))
    a = compute_region_holdout_split(coords, 44, 48, 5, 10.5, 0.5, 0.3, seed=7)
    b = compute_region_holdout_split(coords, 44, 48, 5, 10.5, 0.5, 0.3, seed=7)
    assert a == b


def test_empty_region_raises():
    coords = _coords(out1=(50.0, 7.0))
    with pytest.raises(ValueError, match="no stations"):
        compute_region_holdout_split(coords, 44, 48, 5, 10.5, 0.8, 0.2, seed=0)


def test_degenerate_context_fraction_raises():
    coords = _coords(in1=(45.0, 7.0), in2=(45.5, 7.5), out1=(50.0, 7.0), out2=(51.0, 7.0))
    with pytest.raises(ValueError, match="context_fraction"):
        compute_region_holdout_split(coords, 44, 48, 5, 10.5, 0.0, 0.2, seed=0)


def test_degenerate_val_fraction_raises():
    coords = _coords(in1=(45.0, 7.0), in2=(45.5, 7.5), out1=(50.0, 7.0), out2=(51.0, 7.0))
    with pytest.raises(ValueError, match="val_station_fraction"):
        compute_region_holdout_split(coords, 44, 48, 5, 10.5, 0.8, 0.0, seed=0)


def test_render_config_quotes_numeric_station_ids():
    # Real Meteostat station IDs are digit strings (e.g. "1001"). YAML
    # parses an unquoted "- 1001" as an int, and "- 01001" as octal.
    # render_config must quote every station ID so it round-trips as str.
    content = render_config(
        train_ids=["1001"],
        val_ids=["01001"],
        test_context_ids=["1010"],
        test_eval_ids=["1015"],
        lat_min=44,
        lat_max=48,
        lon_min=5,
        lon_max=10.5,
        context_fraction=0.8,
        val_station_fraction=0.2,
        seed=42,
        station_meta_path="meta.parquet",
        obs_path="obs.parquet",
    )
    loaded = yaml.safe_load(content)
    assert loaded["train_station_indices"] == ["1001"]
    assert isinstance(loaded["train_station_indices"][0], str)
    assert loaded["val_station_indices"] == ["01001"]
    assert isinstance(loaded["val_station_indices"][0], str)
    assert loaded["test_context_station_indices"] == ["1010"]
    assert loaded["test_station_indices"] == ["1015"]
