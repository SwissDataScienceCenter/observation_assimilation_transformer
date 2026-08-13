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
            "transform_type": "keep_support",
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
    ds, _ = _build_dataset()  # transform_type == "real"
    post, output = ds.get_transform(num_data_points=2)
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
    _, output = ds.get_transform(num_data_points=len(ds))
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
    ds, _ = _build_dataset()  # existing helper: obs field on 'stations' coord source
    inner_all, _ = ds.compute_field_leaves(num_data_points=len(ds))
    inner_sub, _ = ds.compute_field_leaves(
        num_data_points=len(ds), target_point_indices=np.array([0])
    )
    # restricting to station 0 changes the obs standardization stats
    obs_name = ds.field_config.roles["target"]
    assert inner_all[obs_name].to_dict() != inner_sub[obs_name].to_dict()


def test_nwp_extent_and_target_coord_source() -> None:
    ds, _ = _build_dataset()
    ext = ds.nwp_extent
    assert set(ext) == {"min", "max"} and ext["min"].shape == (2,)
    assert ds.target_coord_source() == "stations"
