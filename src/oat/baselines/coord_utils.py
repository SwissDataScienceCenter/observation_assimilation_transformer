import numpy as np
import torch
from pyproj import Transformer

_LV95_TO_WGS84: Transformer = Transformer.from_crs(
    "EPSG:2056", "EPSG:4326", always_xy=True
)


def lv95_to_latlon(coords: torch.Tensor) -> torch.Tensor:
    """Convert LV95 (easting, northing) to WGS84 (lat, lon).

    Args:
        coords: (..., 2) tensor [easting, northing] in metres

    Returns:
        (..., 2) tensor [lat, lon] in degrees, same dtype as input
    """
    orig_shape = coords.shape
    flat = coords.reshape(-1, 2).cpu().numpy().astype(np.float64)
    lon, lat = _LV95_TO_WGS84.transform(flat[:, 0], flat[:, 1])
    latlon = np.stack([lat, lon], axis=-1).astype(np.float32)
    return (
        torch.from_numpy(latlon)
        .to(dtype=coords.dtype, device=coords.device)
        .reshape(orig_shape)
    )


def lonlat_swap(coords: torch.Tensor) -> torch.Tensor:
    """Swap (lon, lat) → (lat, lon) for ERA5 station coordinates.

    Args:
        coords: (..., 2) tensor [longitude, latitude]

    Returns:
        (..., 2) tensor [latitude, longitude], same dtype/device
    """
    return torch.stack([coords[..., 1], coords[..., 0]], dim=-1)


def nwp_latlon_grid(
    ref_x: np.ndarray, ref_y: np.ndarray, stride: int = 10
) -> torch.Tensor:
    """Build a coarsened COSMO NWP grid as (lat, lon) coordinates.

    Args:
        ref_x: 1-D easting array, shape (W,)  — e.g. ref_nwp_ds.x.values
        ref_y: 1-D northing array, shape (H,) — e.g. ref_nwp_ds.y.values
        stride: spatial coarsening factor applied to both axes

    Returns:
        (H//stride, W//stride, 2) float32 tensor [lat, lon] per grid point
    """
    x_c = ref_x[::stride]
    y_c = ref_y[::stride]
    grid_x, grid_y = np.meshgrid(x_c, y_c)
    lon, lat = _LV95_TO_WGS84.transform(grid_x.ravel(), grid_y.ravel())
    latlon = np.stack([lat, lon], axis=-1).reshape(len(y_c), len(x_c), 2)
    return torch.from_numpy(latlon.astype(np.float32))


def nwp_latlon_grid_from_latlon(
    ref_lats: np.ndarray, ref_lons: np.ndarray, stride: int = 4
) -> torch.Tensor:
    """Build a coarsened ERA5 NWP grid as (lat, lon) coordinates.

    Args:
        ref_lats: 1-D latitude array, shape (H,) - latitudes of the ERA5 grid
        ref_lons: 1-D longitude array, shape (W,) - longitudes of the ERA5 grid
        stride: spatial coarsening factor applied to both axes

    Returns:
        (H//stride, W//stride, 2) float32 tensor [lat, lon] per grid point
    """
    lats_c = ref_lats[::stride]
    lons_c = ref_lons[::stride]
    grid_lon, grid_lat = np.meshgrid(lons_c, lats_c)
    latlon = np.stack([grid_lat, grid_lon], axis=-1)
    return torch.from_numpy(latlon.astype(np.float32))
