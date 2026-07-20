"""Covariate computation, binning and stats for the error-stratification plot."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

NDArray = np.ndarray[Any, np.dtype[Any]]

SEASONS = ["DJF", "MAM", "JJA", "SON"]
_MONTH_TO_SEASON = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}


def detect_lonlat(coords: NDArray) -> tuple[NDArray, NDArray]:
    """Return (lon, lat) for station coords, auto-detecting the CRS.

    Swiss LV95 (EPSG:2056, easting/northing in metres) is detected when the
    first column exceeds 1e5 and reprojected to WGS84. Otherwise the coords are
    assumed to already be [lon, lat] (era5_meteostat).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if np.abs(coords[:, 0]).max() > 1e5:
        from pyproj import Transformer

        tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(coords[:, 0], coords[:, 1])
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return coords[:, 0], coords[:, 1]


def nearest_obs_distance(eval_coords: NDArray, obs_coords: NDArray) -> NDArray:
    """Min Euclidean distance from each eval station to any observed station.

    Distances are in the native coordinate units (metres for LV95, degrees for
    WGS84).
    """
    eval_coords = np.asarray(eval_coords, dtype=np.float64)
    obs_coords = np.asarray(obs_coords, dtype=np.float64)
    dists = np.linalg.norm(eval_coords[:, None, :] - obs_coords[None, :, :], axis=-1)
    return np.asarray(dists.min(axis=1))


def season_labels(times: Any) -> NDArray:
    """Map a sequence of timestamps to DJF/MAM/JJA/SON labels."""
    months = pd.DatetimeIndex(times).month
    return np.array([_MONTH_TO_SEASON[int(m)] for m in months], dtype=object)


def collect_station_covariates(
    dataset: Any, eval_indices: list[int], obs_indices: list[int]
) -> dict[str, NDArray]:
    """Compute static per-eval-station covariates for the long dump."""
    coords = dataset.station_coords.cpu().numpy()
    eval_coords = coords[eval_indices]
    obs_coords = coords[obs_indices]
    geo = dataset.station_geo.cpu().numpy()  # [N, 3] easting, northing, dem
    lon, lat = detect_lonlat(eval_coords)
    return {
        "station_idx": np.asarray(eval_indices),
        "station_x": eval_coords[:, 0],
        "station_y": eval_coords[:, 1],
        "lon": lon,
        "lat": lat,
        "elevation": geo[eval_indices, 2],
        "nearest_obs_dist": nearest_obs_distance(eval_coords, obs_coords),
    }


def build_long_dataframe(
    *,
    crps: NDArray,
    mae: NDArray,
    times: Any,
    covariates: dict[str, NDArray],
    var_names: list[str],
) -> pd.DataFrame:
    """Assemble a per-(station, timestep) long DataFrame.

    Args:
        crps: [T, S, V] per-(time, station, var) CRPS (NaN where no obs).
        mae: [T, S, V] per-(time, station, var) MAE (NaN where no obs).
        times: length-T sequence of timestamps.
        covariates: output of ``collect_station_covariates`` (length-S arrays).
        var_names: length-V list of variable names.

    Rows are ordered time-major: (t0,s0), (t0,s1), ..., (t1,s0), ...
    """
    T, S, _ = crps.shape
    seasons = season_labels(times)  # [T]
    data: dict[str, NDArray] = {
        "station_idx": np.tile(covariates["station_idx"], T),
        "time": np.repeat(np.asarray(pd.DatetimeIndex(times).values), S),
        "season": np.repeat(seasons, S),
        "station_x": np.tile(covariates["station_x"], T),
        "station_y": np.tile(covariates["station_y"], T),
        "lon": np.tile(covariates["lon"], T),
        "lat": np.tile(covariates["lat"], T),
        "elevation": np.tile(covariates["elevation"], T),
        "nearest_obs_dist": np.tile(covariates["nearest_obs_dist"], T),
    }
    for vi, var in enumerate(var_names):
        data[f"crps_{var}"] = crps[:, :, vi].reshape(-1)
        data[f"mae_{var}"] = mae[:, :, vi].reshape(-1)
    return pd.DataFrame(data)
