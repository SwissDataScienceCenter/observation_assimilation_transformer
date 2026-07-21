import types

import numpy as np
import pandas as pd
import pytest

from oat.plotting_module.stratification import (
    SEASONS,
    build_long_dataframe,
    collect_station_covariates,
    detect_lonlat,
    nearest_obs_distance,
    season_labels,
)


def test_detect_lonlat_lv95_reprojects():
    # Bern-ish LV95 easting/northing (metres) -> ~7.4E, ~46.9N
    coords = np.array([[2600000.0, 1200000.0]])
    lon, lat = detect_lonlat(coords)
    assert 5.0 < lon[0] < 11.0
    assert 45.0 < lat[0] < 48.0


def test_detect_lonlat_wgs84_passthrough():
    coords = np.array([[7.4, 46.9], [8.5, 47.4]])
    lon, lat = detect_lonlat(coords)
    np.testing.assert_allclose(lon, [7.4, 8.5])
    np.testing.assert_allclose(lat, [46.9, 47.4])


def test_nearest_obs_distance():
    # Inputs are WGS84 [lon, lat]; distances are geodesic km. Along a meridian
    # one degree of latitude is ~111.2 km.
    eval_coords = np.array([[0.0, 0.0], [0.0, 10.0]])
    obs_coords = np.array([[0.0, 1.0]])
    d = nearest_obs_distance(eval_coords, obs_coords)
    # eval0 is 1 deg (~111 km) from obs; eval1 is 9 deg (~1001 km).
    np.testing.assert_allclose(d, [111.19, 1000.75], rtol=1e-3)


def test_season_labels():
    times = pd.to_datetime(
        ["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15", "2021-12-15"]
    )
    assert list(season_labels(times)) == ["DJF", "MAM", "JJA", "SON", "DJF"]
    assert set(SEASONS) == {"DJF", "MAM", "JJA", "SON"}


def test_collect_station_covariates():
    import torch

    dataset = types.SimpleNamespace(
        station_coords=torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]),
        station_geo=torch.tensor(
            [[0.0, 0.0, 100.0], [10.0, 0.0, 200.0], [0.0, 10.0, 300.0]]
        ),
    )
    cov = collect_station_covariates(dataset, eval_indices=[1, 2], obs_indices=[0])
    np.testing.assert_array_equal(cov["station_idx"], [1, 2])
    np.testing.assert_allclose(cov["elevation"], [200.0, 300.0])
    # Both eval stations lie 10 deg from the obs at (0,0): ~1112 km geodesic
    # (10 deg along the equator / a meridian).
    np.testing.assert_allclose(cov["nearest_obs_dist"], [1111.95, 1111.95], rtol=1e-3)
    # WGS84 passthrough (small magnitudes)
    np.testing.assert_allclose(cov["lon"], [10.0, 0.0])
    np.testing.assert_allclose(cov["lat"], [0.0, 10.0])


def test_build_long_dataframe_shape_and_nan():
    times = pd.to_datetime(["2021-01-01", "2021-07-01"])  # T=2
    covariates = {
        "station_idx": np.array([5, 6]),
        "station_x": np.array([0.0, 10.0]),
        "station_y": np.array([0.0, 0.0]),
        "lon": np.array([7.0, 8.0]),
        "lat": np.array([46.0, 47.0]),
        "elevation": np.array([100.0, 200.0]),
        "nearest_obs_dist": np.array([1.0, 2.0]),
    }
    crps = np.array([[[0.1], [np.nan]], [[0.2], [0.3]]])  # [T=2, S=2, V=1]
    mae = crps * 2
    df = build_long_dataframe(
        crps=crps,
        mae=mae,
        times=times,
        covariates=covariates,
        var_names=["temperature"],
    )
    assert len(df) == 4  # T*S
    # row ordering is time-major: (t0,s0),(t0,s1),(t1,s0),(t1,s1)
    assert list(df["station_idx"]) == [5, 6, 5, 6]
    assert list(df["season"]) == ["DJF", "DJF", "JJA", "JJA"]
    assert np.isnan(df["crps_temperature"].iloc[1])
    assert df["crps_temperature"].iloc[0] == pytest.approx(0.1)
    assert df["mae_temperature"].iloc[3] == pytest.approx(0.6)
