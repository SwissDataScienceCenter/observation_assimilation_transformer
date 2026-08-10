import pandas as pd
import torch

from oat.data_module.field import Field
from oat.data_module.field_dataset import EPOCH_HOURS, FieldDatasetBase
from oat.data_module.field_spec import parse_field_config
from oat.data_module.sources import PointSetSource, RasterSource
from oat.data_module.support import VariableSupport


def _build_dataset():
    stations = torch.tensor([[0.0, 0.0], [10.0, 10.0]])  # 2 stations
    nwp_coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    # nwp value = station-ish; time-varying: value = base + time_idx
    nwp = PointSetSource(
        nwp_coords,
        ["T_2M"],
        values_fn=lambda t: torch.tensor([[float(t), 1.0 + t, 2.0 + t, 3.0 + t]]),
        native_coord_source="nwp_grid",
    )
    obs = PointSetSource(
        stations,
        ["temperature"],
        values_fn=lambda t: torch.tensor([[20.0 + t, 21.0 + t]]),
        mask_fn=lambda t: torch.tensor([[1.0, 0.0]]),
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
            "transform_type": "real",
        }
    )
    times = pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"])
    support = VariableSupport(
        lower_bounds=torch.tensor([-float("inf")]),
        upper_bounds=torch.tensor([float("inf")]),
        lower_open=torch.tensor([True]),
        upper_open=torch.tensor([True]),
    )
    ds = FieldDatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations, "nwp_grid": nwp_coords},
        sources={"obs": obs, "nwp": nwp, "topo": topo},
        times=times,
        obs_var_support=support,
    )
    return ds, stations


def test_getitem_shapes_and_content():
    ds, stations = _build_dataset()
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
    ds, _ = _build_dataset()
    q = torch.tensor([[0.0, 0.0], [9.0, 9.0]])
    out = ds.get_query(q, idx=0)
    assert set(out) == {"nwp", "geo"}  # obs excluded (not query)
    # query point (9,9) nearest nwp point idx 3 -> value 3.0 (time_idx=0)
    assert out["nwp"].values[1, 0].item() == 3.0
    assert torch.equal(out["geo"].values[:, :2], q)


def test_introspection():
    ds, _ = _build_dataset()
    assert ds.field_coord_sources() == {
        "obs": "stations",
        "nwp": "stations",
        "geo": "stations",
    }
    assert ds.roles()["target"] == "obs"
    assert ds.sources_info()["obs"].has_mask is True
    assert ds.sources_info()["nwp"].available_vars == ["T_2M"]
