"""Zoom area, stations and query locations for the OAT scheme panel plot."""
from __future__ import annotations

import numpy as np

from oat.paths import data_root

# ---------------------------------------------------------------------------
# Zoom parameters
# ---------------------------------------------------------------------------
SEED = 42
ZOOM_CENTER = (2_637_000.0, 1_163_000.0)
ZOOM_HALF_KM = 25_000.0


def zoom_bounds() -> tuple[float, float, float, float]:
    cx, cy = ZOOM_CENTER
    return cx - ZOOM_HALF_KM, cx + ZOOM_HALF_KM, cy - ZOOM_HALF_KM, cy + ZOOM_HALF_KM


def canonical_stations() -> tuple[np.ndarray, np.ndarray]:
    """Return (obs, missing) LV95 coordinates from real PeakWeather stations.

    The obs/missing split is fixed with SEED+1.
    """
    from peakweather.dataset import PeakWeatherDataset

    xlo, xhi, ylo, yhi = zoom_bounds()
    obs_ds = PeakWeatherDataset(
        root=str(data_root() / "peakweather"),
        station_type="meteo_station",
        freq="1h",
    )
    tbl = obs_ds.stations_table
    mask = (
        (tbl["swiss_easting"] >= xlo) & (tbl["swiss_easting"] <= xhi)
        & (tbl["swiss_northing"] >= ylo) & (tbl["swiss_northing"] <= yhi)
    )
    stations = tbl.loc[mask, ["swiss_easting", "swiss_northing"]].values
    rng = np.random.RandomState(SEED + 1)
    missing_idx = rng.randint(len(stations))
    obs_mask = np.ones(len(stations), dtype=bool)
    obs_mask[missing_idx] = False
    return stations[obs_mask], stations[~obs_mask]


def canonical_queries(n: int = 8) -> np.ndarray:
    """Return n fixed query locations (LV95) over the zoom area.

    Uses SEED+2 — independent of the station obs/missing split (SEED+1).
    """
    xlo, xhi, ylo, yhi = zoom_bounds()
    rng = np.random.RandomState(SEED + 2)
    margin = 2_000.0
    pts: list[list[float]] = []
    while len(pts) < n:
        x = rng.uniform(xlo + margin, xhi - margin)
        y = rng.uniform(ylo + margin, yhi - margin)
        pts.append([x, y])
    return np.array(pts)


def canonical_queries_grid() -> np.ndarray:
    """Return a 4×4 regular grid of query locations (LV95) centred in the zoom area.

    Grid spans 5 km × 5 km (±2.5 km from ZOOM_CENTER), giving 16 points ~1.33 km apart.
    """
    cx, cy = ZOOM_CENTER
    # Centre of the SW quadrant: shift −12.5 km in both x and y
    cx_sw = cx - ZOOM_HALF_KM / 2
    cy_sw = cy - ZOOM_HALF_KM / 2
    half = 2_500.0
    xs = np.linspace(cx_sw - half, cx_sw + half, 4)
    ys = np.linspace(cy_sw - half, cy_sw + half, 4)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])
