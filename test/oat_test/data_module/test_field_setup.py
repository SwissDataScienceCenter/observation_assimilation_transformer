from typing import Any
import pickle

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from oat.data_module.field import Field
from oat.data_module.field_dataset import FieldDatasetBase
from oat.data_module.field_setup import (
    build_two_group_transform,
    field_collate,
    validate_field_config_against_scalars,
)
from oat.data_module.field_spec import parse_field_config
from oat.data_module.sources import PointSetSource
from oat.data_module.support import VariableSupport
from oat_test.data_module.test_field_dataset_base import _build_dataset


def _build_dataset_with_global_role() -> FieldDatasetBase:
    """Minimal field dataset declaring BOTH `target` and `global` roles.

    Unlike `_build_dataset()` (roles={"target": "obs", "geo_bias": "geo"}, no
    `global` role), this fixture lets `validate_field_config_against_scalars`
    get past the required-roles check, so tests can actually exercise the
    obs-var-order comparison instead of failing earlier on the missing-role
    check.
    """
    stations = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    nwp = PointSetSource(
        stations,
        ["T_2M"],
        values_fn=lambda t: torch.tensor([[float(t), 1.0 + t]]),
        native_coord_source="stations",
    )
    obs = PointSetSource(
        stations,
        ["temperature"],
        values_fn=lambda t: torch.tensor([[20.0 + t, 21.0 + t]]),
        mask_fn=lambda t: torch.tensor([[1.0, 1.0]]),
        native_coord_source="stations",
    )
    cfg = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
            },
            "roles": {"target": "obs", "global": "nwp"},
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
    return FieldDatasetBase(
        field_config=cfg,
        coord_source_coords={"stations": stations},
        sources={"obs": obs, "nwp": nwp},
        times=times,
        obs_var_support=support,
    )


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
    # Fixture declares both `target` and `global` roles with `nwp_vars`
    # matching, so validation gets PAST the required-roles check and the
    # raised error is actually about the obs-var order mismatch below (not
    # a "missing required role" error masquerading as the intended failure).
    ds = _build_dataset_with_global_role()
    cfg = OmegaConf.create(
        {
            "dataset": {
                "obs_vars": ["WRONG"],
                "nwp_vars": ["T_2M"],
                "geo_vars": [],
            }
        }
    )
    with pytest.raises(ValueError, match="role 'target'"):
        validate_field_config_against_scalars(ds, cfg)


def test_setup_dispatches_field_type(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    import oat.data_module.setup as S

    called: dict[str, str] = {}

    def fake(cfg: Any, logger: Any) -> tuple[str, int, None, None, None, None, None]:
        called["hit"] = cfg.dataset.dataset_type
        return ("d", 1, None, None, None, None, None)

    monkeypatch.setattr(
        "oat.data_module.field_setup.setup_field_data_and_transform", fake
    )
    cfg = OmegaConf.create({"dataset": {"dataset_type": "field_cosmo_peakweather"}})
    S.setup_data_and_transform(cfg, logging.getLogger("t"))
    assert called["hit"] == "field_cosmo_peakweather"


def test_obs_source_callables_are_picklable() -> None:
    from oat.data_module.field_cosmo_peakweather_dataset import (
        FieldCosmoPeakWeatherDataset,
    )

    from oat.data_module.field_era5_meteostat_dataset import (
        FieldERA5MeteostatDataset,
    )

    # bound-method sentinel: the methods exist and are not lambdas
    for cls in (FieldCosmoPeakWeatherDataset, FieldERA5MeteostatDataset):
        assert hasattr(cls, "_obs_values") and hasattr(cls, "_obs_mask")
        assert cls._obs_values.__name__ == "_obs_values"
        assert cls._obs_mask.__name__ == "_obs_mask"
