import functools

import pandas as pd
import torch
from torch import Tensor

from oat.data_module.dataset import EPOCH_HOURS, DatasetBase
from oat.data_module.field import Field
from oat.data_module.field_spec import parse_field_config
from oat.data_module.sources import PointSetSource, RasterSource
from oat.data_module.support import VariableSupport


def _nwp_value_fn(t: int, n_points: int) -> Tensor:
    """nwp value = station-ish; time-varying: value = base + time_idx.

    Module-level (not a local lambda), so it stays picklable when bound via
    ``functools.partial`` for the DDP-worker pickle guard.
    """
    return (torch.arange(n_points, dtype=torch.float32) + t).reshape(1, -1)


def _obs_value_fn(t: int) -> Tensor:
    return torch.tensor([[20.0 + t, 21.0 + t]])


def _obs_mask_fn(t: int, all_valid: bool) -> Tensor:
    return torch.tensor([[1.0, 1.0]] if all_valid else [[1.0, 0.0]])


def _make_dataset(
    all_obs_valid: bool = False,
    coord_source_shapes: dict[str, tuple[int, int]] | None = None,
    nwp_grid_n_points: int | None = None,
):
    """Build a small two-station dataset.

    Set ``all_obs_valid`` to mark both stations valid. The default masks out
    station 1, which the stats gathering skips.

    ``coord_source_shapes`` is forwarded to ``DatasetBase`` unchanged. When it
    names ``"nwp_grid"``, the nwp coord source gets ``nwp_grid_n_points``
    points (default: ``H * W`` from the shape) instead of the usual 4.
    """
    stations = torch.tensor([[0.0, 0.0], [10.0, 10.0]])  # 2 stations
    shapes = coord_source_shapes or {}
    if "nwp_grid" in shapes:
        h, w = shapes["nwp_grid"]
        n_points = nwp_grid_n_points if nwp_grid_n_points is not None else h * w
        nwp_coords = torch.stack(
            [torch.arange(n_points, dtype=torch.float32), torch.zeros(n_points)],
            dim=-1,
        )
    else:
        nwp_coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    nwp = PointSetSource(
        nwp_coords,
        ["T_2M"],
        values_fn=functools.partial(_nwp_value_fn, n_points=nwp_coords.shape[0]),
        native_coord_source="nwp_grid",
    )
    obs = PointSetSource(
        stations,
        ["temperature"],
        values_fn=_obs_value_fn,
        mask_fn=functools.partial(_obs_mask_fn, all_valid=all_obs_valid),
        native_coord_source="stations",
    )
    topo = RasterSource(
        torch.arange(9.0).reshape(1, 3, 3),
        origin=(0.0, 0.0),
        res=(5.0, 5.0),
        var_names=["DEM"],
    )
    cfg = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}, "nwp_grid": {"from": "nwp"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
                "geo": {
                    "coords": "stations",
                    "inputs": ["coords.x", "coords.y", "topo.DEM"],
                    "out": ["e", "n", "dem"],
                    "query": True,
                },
            },
            "roles": {"target": "obs", "geo_bias": "geo"},
        }
    )
    times = pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"])
    support = VariableSupport(
        lower_bounds=torch.tensor([-float("inf")]),
        upper_bounds=torch.tensor([float("inf")]),
        lower_open=torch.tensor([True]),
        upper_open=torch.tensor([True]),
    )
    ds = DatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations, "nwp_grid": nwp_coords},
        sources={"obs": obs, "nwp": nwp, "topo": topo},
        times=times,
        obs_var_support=support,
        coord_source_shapes=coord_source_shapes,
    )
    return ds, stations


def test_getitem_shapes_and_content():
    ds, stations = _make_dataset()
    assert len(ds) == 2
    sample = ds[1]
    fields = sample["fields"]
    assert sample["index"] == 1
    assert set(fields) == {"obs", "nwp", "geo"}
    # obs: identity read, mask from source, coords [N,3] with t column
    assert isinstance(fields["obs"], Field)
    assert fields["obs"].values.shape == (2, 1)
    assert fields["obs"].mask.tolist() == [[1.0], [0.0]]
    assert fields["obs"].coords.shape == (2, 3)
    t_hours = EPOCH_HOURS(pd.to_datetime(["2020-01-01T01:00:00Z"]))[0]
    t_hours = t_hours.to(fields["obs"].coords.dtype)
    assert torch.allclose(fields["obs"].coords[:, 2], t_hours.repeat(2))
    # nwp nearest at stations, time_idx=1 -> station (10,10) is nwp idx 3 = 3.0 + 1
    assert fields["nwp"].values[1, 0].item() == 4.0
    assert fields["nwp"].mask is None
    # geo = concat(x, y, topo.DEM); coords shared by reference with obs/nwp
    assert fields["geo"].var_names == ["e", "n", "dem"]
    assert torch.equal(fields["geo"].values[:, :2], stations)
    assert fields["geo"].coords is fields["obs"].coords  # shared by reference


def test_get_query_only_query_fields_no_obs():
    ds, _ = _make_dataset()
    q = torch.tensor([[0.0, 0.0], [9.0, 9.0]])
    out = ds.get_query(q, idx=0)
    assert set(out) == {"nwp", "geo"}  # obs excluded (not query)
    # query point (9,9) nearest nwp point idx 3 -> value 3.0 (time_idx=0)
    assert out["nwp"].values[1, 0].item() == 3.0
    assert torch.equal(out["geo"].values[:, :2], q)


def _build_dataset_linear():
    stations = torch.tensor([[0.5, 0.0], [2.0, 2.0]])  # 2 stations
    raster = torch.arange(9.0).reshape(1, 3, 3)  # value = row*3 + col
    nwp = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["T"])
    topo = RasterSource(raster, origin=(0.0, 0.0), res=(1.0, 1.0), var_names=["DEM"])
    obs = PointSetSource(
        stations,
        ["temperature"],
        values_fn=lambda t: torch.tensor([[20.0 + t, 21.0 + t]]),
        mask_fn=lambda t: torch.tensor([[1.0, 0.0]]),
        native_coord_source="stations",
    )
    cfg = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {
                    "coords": "stations",
                    "inputs": ["nwp.T"],
                    "query": True,
                    "interp": "linear",
                },
                "geo": {
                    "coords": "stations",
                    "inputs": ["coords.x", "coords.y", "topo.DEM"],
                    "out": ["e", "n", "dem"],
                    "query": True,
                },
            },
            "roles": {"target": "obs", "geo_bias": "geo"},
        }
    )
    times = pd.to_datetime(["2020-01-01T00:00:00Z"])
    support = VariableSupport(
        lower_bounds=torch.tensor([-float("inf")]),
        upper_bounds=torch.tensor([float("inf")]),
        lower_open=torch.tensor([True]),
        upper_open=torch.tensor([True]),
    )
    ds = DatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations},
        sources={"obs": obs, "nwp": nwp, "topo": topo},
        times=times,
        obs_var_support=support,
    )
    return ds, stations, nwp, topo


def test_linear_interpolated_nwp_read_and_nearest_unaffected():
    ds, stations, nwp, topo = _build_dataset_linear()
    fields = ds[0]["fields"]

    # station 0 at (0.5, 0.0): midway between col=0 (value 0) and col=1
    # (value 1) at row=0, bilinear gives mean = 0.5.
    assert abs(fields["nwp"].values[0, 0].item() - 0.5) < 1e-6
    # station 1 sits exactly on grid point (2, 2) -> value 8.
    assert abs(fields["nwp"].values[1, 0].item() - 8.0) < 1e-6

    # nearest field (geo's topo.DEM) is unaffected: matches build_index.
    idx = topo.build_index(stations)
    expected_dem = topo.load(0)[0][idx]
    assert torch.equal(fields["geo"].values[:, 2], expected_dem)


def test_get_query_linear_interpolated_nwp_read():
    ds, stations, _, _ = _build_dataset_linear()
    out = ds.get_query(stations, idx=0)
    assert abs(out["nwp"].values[0, 0].item() - 0.5) < 1e-6
    assert abs(out["nwp"].values[1, 0].item() - 8.0) < 1e-6


def test_introspection():
    ds, _ = _make_dataset()
    assert ds.field_coord_sources() == {
        "obs": "stations",
        "nwp": "stations",
        "geo": "stations",
    }
    assert ds.roles()["target"] == "obs"
    assert ds.sources_info()["obs"].has_mask is True
    assert ds.sources_info()["nwp"].available_vars == ["T_2M"]


def test_project_index_coords_default_is_identity():
    import torch
    from oat.data_module.dataset import DatasetBase

    coords = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    # Call the unbound default directly — no instance construction needed.
    out = DatasetBase.project_index_coords(object.__new__(DatasetBase), coords)
    assert out.shape == coords.shape
    assert torch.equal(out, coords)


def test_grid_shape_returns_shape_for_grid_coord_source():
    """grid_shape returns (H, W) for a coord source with a recorded shape."""
    ds, _ = _make_dataset(
        coord_source_shapes={"nwp_grid": (3, 4)},
    )
    assert ds.grid_shape("nwp_grid") == (3, 4)


def test_grid_shape_returns_none_for_station_coord_source():
    """grid_shape returns None for coord sources without a recorded shape."""
    ds, _ = _make_dataset()
    assert ds.grid_shape("stations") is None


def test_grid_shape_asserts_point_count():
    """grid_shape construction fails when H*W != coord point count."""
    import pytest

    with pytest.raises(AssertionError, match="12 points but _coords has"):
        _make_dataset(
            coord_source_shapes={"nwp_grid": (3, 4)},
            nwp_grid_n_points=10,  # 10 != 3*4=12
        )


def test_pickle_roundtrip_with_grid_field():
    """Dataset with a grid-coord field survives pickle (DDP worker guard)."""
    import pickle

    ds, _ = _make_dataset(coord_source_shapes={"nwp_grid": (3, 4)})
    data = pickle.dumps(ds)
    assert len(data) > 0
