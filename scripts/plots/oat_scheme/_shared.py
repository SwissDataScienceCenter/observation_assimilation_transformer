"""Shared constants and data loaders for OAT scheme panel plots.

All panels must import canonical_stations() and canonical_queries() from here
so that every spatial plot shows the same observed stations, missing
observation, and query locations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from matplotlib.colors import LinearSegmentedColormap

from oat.paths import data_root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_COSMO_DIR = data_root() / "cosmo-1e_rechunked"
TOPO_PATH = _COSMO_DIR / "topo.zarr"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
ORANGE = "#E8732A"
GRAY_STA = "#AAAAAA"
BLUE_Q = "#3A72C6"
CMAP_FIELD = "RdBu_r"
FIELD_VLIM = (-3.5, 3.5)
CMAP_UNC = "YlOrBr"

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "terrain_oat",
    [
        (0.00, "#3B5E1F"),
        (0.12, "#5C8A3A"),
        (0.28, "#8EAA62"),
        (0.42, "#C8B57A"),
        (0.55, "#96703C"),
        (0.68, "#7A6050"),
        (0.80, "#A09898"),
        (0.90, "#D2D2D2"),
        (1.00, "#F4F4F4"),
    ],
)

# ---------------------------------------------------------------------------
# Zoom parameters
# ---------------------------------------------------------------------------
SEED = 42
STEP_ZOOM = 2
ZOOM_CENTER = (2_637_000.0, 1_163_000.0)
ZOOM_HALF_KM = 25_000.0


def zoom_bounds() -> tuple[float, float, float, float]:
    cx, cy = ZOOM_CENTER
    return cx - ZOOM_HALF_KM, cx + ZOOM_HALF_KM, cy - ZOOM_HALF_KM, cy + ZOOM_HALF_KM


def load_topo_zoom() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dem, x_1d, y_1d) cropped to zoom window at STEP_ZOOM resolution."""
    xlo, xhi, ylo, yhi = zoom_bounds()
    z = zarr.open(str(TOPO_PATH), mode="r")
    x_full = z["x"][:]
    y_full = z["y"][:]
    xi0, xi1 = int(np.searchsorted(x_full, xlo)), int(np.searchsorted(x_full, xhi))
    yi0, yi1 = int(np.searchsorted(y_full, ylo)), int(np.searchsorted(y_full, yhi))
    dem = z["DEM"][yi0:yi1:STEP_ZOOM, xi0:xi1:STEP_ZOOM]
    x_1d = x_full[xi0:xi1:STEP_ZOOM]
    y_1d = y_full[yi0:yi1:STEP_ZOOM]
    return dem.astype(np.float32), x_1d.astype(np.float64), y_1d.astype(np.float64)


def canonical_stations() -> tuple[np.ndarray, np.ndarray]:
    """Return (obs, missing) LV95 coordinates from real PeakWeather stations.

    The obs/missing split is fixed via SEED+1 and is identical across all panels.
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
