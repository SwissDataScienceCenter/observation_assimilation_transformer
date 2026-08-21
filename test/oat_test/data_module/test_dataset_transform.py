import numpy as np
import pandas as pd
import torch

from oat.data_module.dataset import DatasetBase
from oat.data_module.field_spec import parse_field_config
from oat.data_module.sources import PointSetSource
from oat.data_module.support import VariableSupport
from oat.data_module.transform import (
    ComposedTransform,
    FieldValueTransform,
    NestedTransform,
    TranslateScaleTransform,
)
from oat_test.data_module.test_dataset_base import _build_dataset


def _build_keep_support_dataset() -> DatasetBase:
    """Obs field with two channels of mixed support at the lower bound zero.

    Var "A" is strictly positive (lower_open=True, e.g. wind speed) and MUST
    get the zero-avoiding offset before standardization. Var "B" is closed at
    zero (lower_open=False, e.g. precipitation) and MUST NOT be offset, since
    that would corrupt its legitimate zero-inflation point.
    """
    stations = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    obs = PointSetSource(
        stations,
        ["A", "B"],
        values_fn=lambda t: torch.tensor([[5.0 + t, 6.0 + t], [3.0 + t, 4.0 + t]]),
        mask_fn=lambda t: torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        native_coord_source="stations",
    )
    cfg = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.A", "obs.B"]},
            },
            "roles": {"target": "obs"},
        }
    )
    times = pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"])
    support = VariableSupport(
        lower_bounds=torch.tensor([0.0, 0.0]),
        upper_bounds=torch.tensor([float("inf"), float("inf")]),
        lower_open=torch.tensor([True, False]),
        upper_open=torch.tensor([True, True]),
    )
    return DatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations},
        sources={"obs": obs},
        times=times,
        obs_var_support=support,
    )


def test_get_transform_real_standardizes_all_and_logs_obs():
    ds, _ = _build_dataset()  # type defaults to "real"
    post, output = ds.get_transform(num_data_points=2, type="real")
    assert isinstance(post, NestedTransform)
    fields_tr = post.transform_dict["fields"].transform_dict
    # every field has a FieldValueTransform leaf
    assert set(fields_tr.keys()) == {"obs", "nwp", "geo"}
    for leaf in fields_tr.values():
        assert isinstance(leaf, FieldValueTransform)
    # applying to a batch standardizes values and preserves coords/mask
    sample = ds[0]
    out = post({"fields": sample["fields"]})
    for name, f in out["fields"].items():
        assert f.coords is sample["fields"][name].coords
    # output_transform round-trips obs values
    obs_vals = sample["fields"]["obs"].values
    normed = output(obs_vals)
    assert torch.allclose(output.inverse(normed), obs_vals, atol=1e-4)


def test_get_transform_keep_support_offsets_only_strictly_positive_var():
    ds = _build_keep_support_dataset()
    _, output = ds.get_transform(num_data_points=len(ds), type="keep_support")
    assert isinstance(output, ComposedTransform)
    obs_pre = output.transforms[0]
    assert isinstance(obs_pre, TranslateScaleTransform)
    # var "A" (strictly positive, lower_open=True) is offset away from zero;
    # var "B" (closed at zero, lower_open=False) is left untouched.
    assert obs_pre.translate[0].item() != 0.0
    assert obs_pre.translate[1].item() == 0.0
    # round-trips real obs values through the composed obs transform
    sample = ds[0]
    obs_vals = sample["fields"]["obs"].values
    normed = output(obs_vals)
    assert torch.allclose(output.inverse(normed), obs_vals, atol=1e-4)


def test_target_point_indices_restricts_stats() -> None:
    # Both stations must be valid, else the stats gathering already skips
    # station 1 and restricting to station 0 changes nothing.
    ds, _ = _build_dataset(all_obs_valid=True)
    inner_all, _ = ds.compute_field_leaves(num_data_points=len(ds), type="real")
    inner_sub, _ = ds.compute_field_leaves(
        num_data_points=len(ds), target_point_indices=np.array([0]), type="real"
    )
    # restricting to station 0 changes the obs standardization stats
    obs_name = ds.field_config.roles["target"]
    assert inner_all[obs_name].to_dict() != inner_sub[obs_name].to_dict()


def test_nwp_extent_and_target_coord_source() -> None:
    ds, _ = _build_dataset()
    ext = ds.nwp_extent
    assert set(ext) == {"min", "max"} and ext["min"].shape == (2,)
    assert ds.target_coord_source() == "stations"


def _build_masked_obs_dataset() -> DatasetBase:
    """Obs field where station 1 is always masked out and holds a filler zero.

    The dataset writes an exact zero wherever the mask is 0 (PeakWeather
    zero-imputes, and `_obs_values` applies `nan_to_num`). Those fillers must
    not enter the normalization statistics.
    """
    stations = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    obs = PointSetSource(
        stations,
        ["A"],
        # [C, N]: station 0 is valid (4.0 then 6.0), station 1 is a masked zero
        values_fn=lambda t: torch.tensor([[4.0 + 2.0 * t, 0.0]]),
        mask_fn=lambda t: torch.tensor([[1.0, 0.0]]),
        native_coord_source="stations",
    )
    cfg = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {"obs": {"coords": "stations", "inputs": ["obs.A"]}},
            "roles": {"target": "obs"},
        }
    )
    times = pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"])
    # unbounded support, so the "real" pre-transform leaves the values alone
    support = VariableSupport(
        lower_bounds=torch.tensor([-float("inf")]),
        upper_bounds=torch.tensor([float("inf")]),
        lower_open=torch.tensor([True]),
        upper_open=torch.tensor([True]),
    )
    return DatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations},
        sources={"obs": obs},
        times=times,
        obs_var_support=support,
    )


def test_stats_ignore_masked_out_entries() -> None:
    """Masked-out fillers must not shift the obs mean/std.

    Valid values are 4.0 and 6.0, so mean=5.0 and std=1.0. Counting the two
    masked zeros instead gives mean=2.5 and std=2.598.
    """
    ds = _build_masked_obs_dataset()
    _, output = ds.compute_field_leaves(num_data_points=len(ds), type="real")
    # the standardization must be centred on the valid values only
    assert torch.allclose(output(torch.tensor([[5.0]])), torch.zeros(1, 1), atol=1e-4)
    assert torch.allclose(output(torch.tensor([[6.0]])), torch.ones(1, 1), atol=1e-4)
