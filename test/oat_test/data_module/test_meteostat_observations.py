import numpy as np
import pandas as pd
import pytest
from oat.data_module.meteostat_observations import MeteostatObservations


def _make_obs_parquet(path):
    times = pd.date_range("2020-01-01", periods=4, freq="1h")
    rows = []
    for station in ["A001", "A002"]:
        for t in times:
            rows.append({
                "station": station, "time": str(t),
                "temp": 10.0, "rhum": 70.0, "prcp": 0.0,
                "wspd": 5.0, "wpgt": float("nan"),
            })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _make_meta_parquet(path):
    pd.DataFrame({
        "station": ["A001", "A002", "A003"],
        "latitude": [47.0, 48.0, float("nan")],
        "longitude": [8.0, 9.0, float("nan")],
        "elevation": [500.0, 600.0, float("nan")],
        "tier": ["wmo", "wmo", "other"],
        "country": ["CH", "DE", "XX"],
    }).to_parquet(path, index=False)


@pytest.fixture
def obs_store(tmp_path):
    _make_obs_parquet(tmp_path / "obs.parquet")
    _make_meta_parquet(tmp_path / "meta.parquet")
    return MeteostatObservations(
        obs_path=str(tmp_path / "obs.parquet"),
        station_meta_path=str(tmp_path / "meta.parquet"),
    )


def test_stations_are_string_array(obs_store):
    s = obs_store.stations
    assert isinstance(s, np.ndarray)
    assert s.dtype.kind in ("U", "O")
    assert set(s) == {"A001", "A002"}  # A003 has no coordinates


def test_stations_table_columns_and_index(obs_store):
    tbl = obs_store.stations_table
    assert tbl.index.name == "station"
    for col in ("latitude", "longitude", "elevation"):
        assert col in tbl.columns
    assert "A001" in tbl.index
    assert "A003" not in tbl.index


def test_get_observations_shape(obs_store):
    obs, mask = obs_store.get_observations(
        first_date=pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        last_date=pd.Timestamp("2020-01-01 03:00", tz="UTC"),
        parameters=["temp", "wspd"],
        stations=np.array(["A001", "A002"]),
        return_mask=True, as_numpy=True,
    )
    assert obs.shape == (4, 2, 2)
    assert mask.shape == (4, 2, 2)
    assert obs.dtype == np.float32


def test_get_observations_nan_yields_false_mask(obs_store):
    obs, mask = obs_store.get_observations(
        first_date=pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        last_date=pd.Timestamp("2020-01-01 03:00", tz="UTC"),
        parameters=["wpgt"],
        stations=np.array(["A001"]),
        return_mask=True, as_numpy=True,
    )
    assert not mask.any()


def test_get_observations_unknown_station(obs_store):
    obs, mask = obs_store.get_observations(
        first_date=pd.Timestamp("2020-01-01 00:00", tz="UTC"),
        last_date=pd.Timestamp("2020-01-01 01:00", tz="UTC"),
        parameters=["temp"],
        stations=np.array(["UNKNOWN"]),
        return_mask=True, as_numpy=True,
    )
    assert obs.shape == (2, 1, 1)
    assert not mask.any()
