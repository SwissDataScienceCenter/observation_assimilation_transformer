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


def build_land_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return boolean mask (len(ys) × len(xs)), True where the grid point is on land.

    Uses Natural Earth 10 m land polygons minus 10 m lake polygons.
    """
    import shapely
    from cartopy.feature import NaturalEarthFeature
    from pyproj import Transformer

    x_grid, y_grid = np.meshgrid(xs, ys)
    is_lv95 = float(np.abs(xs).max()) > 1e5

    if is_lv95:
        tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
        lons_flat, lats_flat = tf.transform(x_grid.ravel(), y_grid.ravel())
        lons_flat, lats_flat = np.asarray(lons_flat), np.asarray(lats_flat)
    else:
        lons_flat, lats_flat = x_grid.ravel().copy(), y_grid.ravel().copy()

    if "land_geom" not in _GEO_CACHE:
        land_arr = np.array(
            list(NaturalEarthFeature("physical", "land", "10m").geometries()),
            dtype=object,
        )
        lakes_arr = np.array(
            list(NaturalEarthFeature("physical", "lakes", "10m").geometries()),
            dtype=object,
        )
        _GEO_CACHE["land_geom"] = shapely.difference(
            shapely.union_all(land_arr),
            shapely.union_all(lakes_arr),
        )

    clipped = shapely.clip_by_rect(
        _GEO_CACHE["land_geom"],
        float(lons_flat.min()) - 1.0,
        float(lats_flat.min()) - 1.0,
        float(lons_flat.max()) + 1.0,
        float(lats_flat.max()) + 1.0,
    )
    shapely.prepare(clipped)
    return shapely.contains_xy(clipped, lons_flat, lats_flat).reshape(x_grid.shape)


def build_domain_grid(
    dataset: Any,
    margin: float = 0.05,
    resolution: float | None = None,
    center: tuple[float, float] | None = None,
    extent: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a query grid, clamped to the region where topography is finite.

    Default box spans the station bounding box plus a fractional *margin*. When
    *center* and *extent* are given, the box is ``center ± extent`` instead.
    Either way it is clamped to the finite-topo region (derived topo channels
    have NaN near the southern boundary). Default *resolution* targets ~100 grid
    points across the domain width.

    Args:
        dataset: Exposes ``nwp_extent``, ``topo_extent``, ``topo_grid``,
            ``topo_field`` (C, H, W), ``station_coords`` (N, 2).
        margin: Fractional margin around the station bounding box.
        resolution: Grid spacing; default domain_width / 100.
        center: Optional (E, N) grid centre overriding the station bbox.
        extent: Optional half-width (same units) used with *center*.

    Returns:
        (xs, ys) 1-D coordinate arrays.
    """
    import torch

    ext = dataset.nwp_extent
    topo_ext = dataset.topo_extent
    patch_margin = dataset.topo_grid.abs().max().item()
    H, W = dataset.topo_field.shape[1:]
    topo_dx = (topo_ext["max"][0].item() - topo_ext["min"][0].item()) / (W - 1)
    topo_dy = (topo_ext["max"][1].item() - topo_ext["min"][1].item()) / (H - 1)
    col0 = max(0, int((ext["min"][0].item() - topo_ext["min"][0].item()) / topo_dx))
    col1 = min(
        W - 1, int((ext["max"][0].item() - topo_ext["min"][0].item()) / topo_dx) + 1
    )
    row0 = max(0, int((ext["min"][1].item() - topo_ext["min"][1].item()) / topo_dy))
    row1 = min(
        H - 1, int((ext["max"][1].item() - topo_ext["min"][1].item()) / topo_dy) + 1
    )
    stride = 10
    sub = dataset.topo_field[1:, row0:row1:stride, col0:col1:stride]
    row_valid = torch.isfinite(sub).all(dim=0).all(dim=1)
    valid_r = row_valid.nonzero(as_tuple=True)[0]
    extra = stride * topo_dy + patch_margin
    if len(valid_r):
        topo_y_min = (
            topo_ext["min"][1].item()
            + (row0 + valid_r[0].item() * stride) * topo_dy
            + extra
        )
        topo_y_max = (
            topo_ext["min"][1].item()
            + (row0 + valid_r[-1].item() * stride) * topo_dy
            - extra
        )
    else:
        topo_y_min, topo_y_max = ext["min"][1].item(), ext["max"][1].item()
    topo_x_min = topo_ext["min"][0].item() + patch_margin
    topo_x_max = topo_ext["max"][0].item() - patch_margin

    if center is not None and extent is not None:
        cx, cy = float(center[0]), float(center[1])
        x_min0, x_max0 = cx - extent, cx + extent
        y_min0, y_max0 = cy - extent, cy + extent
    else:
        st_coords = dataset.station_coords.numpy()
        st_x_min, st_y_min = float(st_coords[:, 0].min()), float(st_coords[:, 1].min())
        st_x_max, st_y_max = float(st_coords[:, 0].max()), float(st_coords[:, 1].max())
        margin_x = (st_x_max - st_x_min) * margin
        margin_y = (st_y_max - st_y_min) * margin
        x_min0, x_max0 = st_x_min - margin_x, st_x_max + margin_x
        y_min0, y_max0 = st_y_min - margin_y, st_y_max + margin_y

    x_min = max(x_min0, topo_x_min)
    x_max = min(x_max0, topo_x_max)
    y_min = max(y_min0, topo_y_min)
    y_max = min(y_max0, topo_y_max)

    domain_width = x_max - x_min
    res: float = resolution if resolution is not None else domain_width / 100.0
    xs = np.arange(x_min, x_max + 1e-9, res)
    ys = np.arange(y_min, y_max + 1e-9, res)
    return xs, ys
