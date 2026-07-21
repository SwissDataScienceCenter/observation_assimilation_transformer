import numpy as np
import torch

from oat.baselines.coord_utils import (
    lv95_to_latlon,
    lonlat_swap,
    nwp_latlon_grid,
    nwp_latlon_grid_from_latlon,
)


def test_lv95_to_latlon_bern():
    coords = torch.tensor([[2600000.0, 1200000.0]])
    result = lv95_to_latlon(coords)
    assert result.shape == (1, 2)
    lat, lon = result[0, 0].item(), result[0, 1].item()
    assert abs(lat - 46.95) < 0.1, f"lat {lat:.4f} not within 0.1 of 46.95"
    assert abs(lon - 7.44) < 0.1, f"lon {lon:.4f} not within 0.1 of 7.44"


def test_lv95_to_latlon_dtype_preserved():
    coords = torch.tensor([[2600000.0, 1200000.0]], dtype=torch.float32)
    assert lv95_to_latlon(coords).dtype == torch.float32


def test_lv95_to_latlon_batch_shape():
    coords = torch.rand(5, 3, 2) * 1e5 + torch.tensor([2.5e6, 1.1e6])
    assert lv95_to_latlon(coords).shape == (5, 3, 2)


def test_lonlat_swap():
    coords = torch.tensor([[7.44, 46.95], [8.54, 47.37]])  # [lon, lat]
    result = lonlat_swap(coords)
    assert result.shape == (2, 2)
    assert torch.allclose(result[:, 0], coords[:, 1])
    assert torch.allclose(result[:, 1], coords[:, 0])


def test_lonlat_swap_batch():
    coords = torch.rand(3, 5, 2)
    swapped = lonlat_swap(coords)
    assert torch.allclose(swapped[..., 0], coords[..., 1])
    assert torch.allclose(swapped[..., 1], coords[..., 0])


def test_nwp_latlon_grid_shape():
    ref_x = np.linspace(2255500.0, 2964500.0, 710)
    ref_y = np.linspace(840500.0, 1479500.0, 640)
    grid = nwp_latlon_grid(ref_x, ref_y, stride=10)
    assert grid.shape == (64, 71, 2)


def test_nwp_latlon_grid_switzerland_bounds():
    ref_x = np.linspace(2255500.0, 2964500.0, 710)
    ref_y = np.linspace(840500.0, 1479500.0, 640)
    grid = nwp_latlon_grid(ref_x, ref_y, stride=10)
    assert grid[..., 0].min() > 43.0 and grid[..., 0].max() < 50.0
    assert grid[..., 1].min() > 2.0 and grid[..., 1].max() < 13.0


def test_nwp_latlon_grid_float32():
    ref_x = np.linspace(2255500.0, 2964500.0, 710)
    ref_y = np.linspace(840500.0, 1479500.0, 640)
    assert nwp_latlon_grid(ref_x, ref_y, stride=10).dtype == torch.float32


def test_nwp_latlon_grid_from_latlon_shape():
    ref_lats = np.arange(34.0, 71.75 + 1e-9, 0.25)   # 152 points
    ref_lons = np.arange(-24.75, 45.0 + 1e-9, 0.25)  # 280 points
    grid = nwp_latlon_grid_from_latlon(ref_lats, ref_lons, stride=4)
    assert grid.shape == (38, 70, 2)


def test_nwp_latlon_grid_from_latlon_values():
    ref_lats = np.arange(34.0, 71.75 + 1e-9, 0.25)
    ref_lons = np.arange(-24.75, 45.0 + 1e-9, 0.25)
    grid = nwp_latlon_grid_from_latlon(ref_lats, ref_lons, stride=4)
    assert abs(grid[0, 0, 0].item() - 34.0) < 0.01
    assert abs(grid[0, 0, 1].item() - (-24.75)) < 0.01
    assert grid.dtype == torch.float32
