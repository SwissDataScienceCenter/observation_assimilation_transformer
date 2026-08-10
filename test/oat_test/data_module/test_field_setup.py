from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from oat.data_module.field import Field
from oat.data_module.field_setup import (
    build_two_group_transform,
    field_collate,
    validate_field_config_against_scalars,
)
from oat_test.data_module.test_field_dataset_base import _build_dataset


def _sample(idx: int, mask: bool = True) -> dict[str, Any]:
    def mk(n: int) -> Field:
        c = torch.zeros(n, 3)
        v = torch.arange(float(n)).reshape(n, 1) + idx
        m = torch.ones(n, 1) if mask else None
        return Field(c, v, ["temperature"], m)

    return {
        "index": idx,
        "observation_data": {"fields": {"obs": mk(2)}},
        "query_data": {"fields": {"obs": mk(1)}},
    }


def test_field_collate_stacks_to_batched_fields() -> None:
    batch = field_collate([_sample(0), _sample(1)])
    obs = batch["observation_data"]["fields"]["obs"]
    assert obs.coords.shape == (2, 2, 3) and obs.values.shape == (2, 2, 1)
    assert obs.mask is not None and obs.mask.shape == (2, 2, 1)
    assert batch["query_data"]["fields"]["obs"].values.shape == (2, 1, 1)
    assert batch["index"].tolist() == [0, 1]


def test_field_collate_none_mask() -> None:
    batch = field_collate([_sample(0, mask=False), _sample(1, mask=False)])
    assert batch["observation_data"]["fields"]["obs"].mask is None


def test_build_two_group_transform_applies_both_groups() -> None:
    ds, _ = _build_dataset()
    inner, _output = ds.compute_field_leaves(num_data_points=len(ds))
    tr = build_two_group_transform(inner)
    # structure: observation_data.fields.<name> and query_data.fields.<name>
    sample = ds[0]["fields"]
    batch = {
        "observation_data": {"fields": sample},
        "query_data": {"fields": sample},
    }
    normed = tr(batch)
    assert "fields" in normed["observation_data"]
    assert "fields" in normed["query_data"]
    # each group's per-field transform is an independent deep copy: same
    # serialized content, but never the same nn.Module object (no shared
    # module registered under both ModuleDict paths).
    obs_group = tr.transform_dict["observation_data"].transform_dict["fields"]
    qry_group = tr.transform_dict["query_data"].transform_dict["fields"]
    obs_leaves = obs_group.transform_dict
    qry_leaves = qry_group.transform_dict
    assert obs_leaves.keys() == qry_leaves.keys() == inner.keys()
    for name in obs_leaves:
        assert obs_leaves[name] is not qry_leaves[name]
        assert obs_leaves[name].to_dict() == qry_leaves[name].to_dict()


def test_validation_rejects_obs_order_mismatch() -> None:
    ds, _ = _build_dataset()
    cfg = OmegaConf.create(
        {
            "dataset": {
                "obs_vars": ["WRONG"],
                "nwp_vars": ["T_2M"],
                "geo_vars": [],
            }
        }
    )
    with pytest.raises(ValueError):
        validate_field_config_against_scalars(ds, cfg)
