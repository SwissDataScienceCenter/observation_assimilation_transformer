"""Shared geo-overlay helpers (coastlines, country borders, lake outlines).

Results are cached in _GEO_CACHE so repeated calls within a session do not
re-download or re-compute Natural Earth data.

Coordinate systems are auto-detected: Swiss LV95 (EPSG:2056) when
max|xs| > 1e5, otherwise WGS84 lon/lat.
"""

from typing import Any

import numpy as np

_GEO_CACHE: dict[str, Any] = {}


def _iter_line_coords(geom: Any) -> list[np.ndarray]:
    """Return (N, 2) coordinate arrays for all linear parts of a shapely geometry."""
    gt = geom.geom_type
    if gt in ("LineString", "LinearRing"):
        return [np.array(geom.coords)]
    if gt == "MultiLineString":
        return [np.array(p.coords) for p in geom.geoms]
    if gt == "Polygon":
        return [np.array(geom.exterior.coords)] + [
            np.array(r.coords) for r in geom.interiors
        ]
    if gt == "MultiPolygon":
        out: list[np.ndarray] = []
        for part in geom.geoms:
            out.append(np.array(part.exterior.coords))
            out.extend(np.array(r.coords) for r in part.interiors)
        return out
    if gt == "GeometryCollection":
        out = []
        for part in geom.geoms:
            out.extend(_iter_line_coords(part))
        return out
    return []


def _add_borders(data_axes: list[Any], xs: np.ndarray, ys: np.ndarray) -> None:
    """Overlay fine borders (coastlines, country lines, lake outlines) on *data_axes*.

    Uses Natural Earth 10 m features. Results are cached after the first call.
    """
    import shapely
    from cartopy.feature import NaturalEarthFeature
    from pyproj import Transformer

    is_lv95 = float(np.abs(xs).max()) > 1e5
    x_grid, y_grid = np.meshgrid(xs, ys)

    if is_lv95:
        tf_to_wgs = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lons_flat, lats_flat = tf_to_wgs.transform(x_grid.ravel(), y_grid.ravel())
        lons_flat, lats_flat = np.asarray(lons_flat), np.asarray(lats_flat)
        tf_to_proj: Any = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    else:
        lons_flat, lats_flat = x_grid.ravel(), y_grid.ravel()
        tf_to_proj = None

    lon_min = float(lons_flat.min()) - 0.5
    lon_max = float(lons_flat.max()) + 0.5
    lat_min = float(lats_flat.min()) - 0.5
    lat_max = float(lats_flat.max()) + 0.5

    cache_key = f"borders_{lon_min:.2f}_{lat_min:.2f}_{lon_max:.2f}_{lat_max:.2f}"
    if cache_key not in _GEO_CACHE:
        feat_defs = [
            ("physical", "coastline"),
            ("cultural", "admin_0_boundary_lines_land"),
            ("physical", "lakes"),
        ]
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for category, name in feat_defs:
            for geom in NaturalEarthFeature(category, name, "10m").geometries():
                clipped = shapely.clip_by_rect(geom, lon_min, lat_min, lon_max, lat_max)
                if clipped.is_empty:
                    continue
                for coords in _iter_line_coords(clipped):
                    if tf_to_proj is not None:
                        xb, yb = tf_to_proj.transform(coords[:, 0], coords[:, 1])
                    else:
                        xb, yb = coords[:, 0], coords[:, 1]
                    segments.append((np.asarray(xb), np.asarray(yb)))
        _GEO_CACHE[cache_key] = segments

    for ax in data_axes:
        for xb, yb in _GEO_CACHE[cache_key]:
            ax.plot(xb, yb, color="black", linewidth=0.4, zorder=3)
