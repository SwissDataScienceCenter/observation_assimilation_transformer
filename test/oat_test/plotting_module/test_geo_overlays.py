from unittest.mock import MagicMock

import numpy as np
import torch

from oat.plotting_module.geo_overlays import build_domain_grid


def _make_grid_dataset():
    ds = MagicMock()
    ds.nwp_extent = {
        "min": torch.tensor([10.0, 10.0]),
        "max": torch.tensor([90.0, 90.0]),
    }
    ds.topo_extent = {
        "min": torch.tensor([0.0, 0.0]),
        "max": torch.tensor([100.0, 100.0]),
    }
    ds.topo_grid = torch.zeros(1, 2)  # patch_margin = 0
    ds.topo_field = torch.zeros(2, 101, 101)  # all finite → every row valid
    ds.station_coords = torch.tensor([[30.0, 30.0], [70.0, 70.0], [50.0, 50.0]])
    return ds


def test_build_domain_grid_monotonic_nonempty():
    xs, ys = build_domain_grid(_make_grid_dataset())
    assert len(xs) > 1 and len(ys) > 1
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)


def test_build_domain_grid_clamps_to_valid_topo():
    # stride=10, topo_dy=1.0, patch_margin=0 → extra=10 → valid y in [20, 80].
    _, ys = build_domain_grid(_make_grid_dataset())
    assert ys.min() >= 20.0 - 1e-6
    assert ys.max() <= 80.0 + 1e-6


def test_build_domain_grid_respects_resolution():
    xs, _ = build_domain_grid(_make_grid_dataset(), resolution=5.0)
    assert abs((xs[1] - xs[0]) - 5.0) < 1e-6


def test_build_domain_grid_center_extent_override():
    # center/extent box (48±5 = [43,53]) is tighter than the station bbox.
    xs, ys = build_domain_grid(
        _make_grid_dataset(), resolution=1.0, center=(48.0, 48.0), extent=5.0
    )
    assert xs.min() >= 43.0 - 1e-6 and xs.max() <= 53.0 + 1e-6
    assert ys.min() >= 43.0 - 1e-6 and ys.max() <= 53.0 + 1e-6
