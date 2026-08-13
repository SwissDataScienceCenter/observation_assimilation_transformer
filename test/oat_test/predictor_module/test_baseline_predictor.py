# test/oat_test/predictor_module/test_baseline_predictor.py
import types
from typing import Any

import pytest
import torch

from oat.data_module.field import Field
from oat.data_module.field_spec import parse_field_config
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.predictor_module.baseline_predictor import (
    NWPInterpolatorPredictor,
    create_baseline_predictor,
)

FIELD_ROLES = {"target": "obs", "global": "nwp", "geo_bias": "geo", "topo": "topo"}


def _make_batch(n: int, t_2m_values: list[float]) -> dict[str, Any]:
    coords = torch.zeros(1, n, 3)
    coords[..., 2] = 100.0  # epoch hours
    nwp_vals = torch.tensor(t_2m_values).reshape(1, n, 1)
    query_fields = {
        "obs": Field(
            coords.clone(), torch.zeros(1, n, 1), ["temperature"], torch.ones(1, n, 1)
        ),
        "nwp": Field(coords.clone(), nwp_vals, ["T_2M"]),
        "geo": Field(coords.clone(), torch.zeros(1, n, 3), ["x", "y", "d"]),
        "topo": Field(coords.clone(), torch.zeros(1, n, 2), ["a", "b"]),
    }
    return {"query_data": {"fields": query_fields}}


@pytest.fixture
def config() -> NWPInterpolatorConfig:
    return NWPInterpolatorConfig(
        nwp_names=["T_2M"],
        obs_names=["temperature"],
        obs_to_nwp={"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)},
    )


@pytest.fixture
def predictor(config: NWPInterpolatorConfig) -> NWPInterpolatorPredictor:
    return NWPInterpolatorPredictor(
        config=config, output_var_names=["temperature"], field_roles=FIELD_ROLES
    )


def test_predict_maps_nwp_to_obs_units(predictor: NWPInterpolatorPredictor) -> None:
    batch = _make_batch(2, [300.0, 310.0])
    normalized = predictor._normalize_batch(batch)
    y_pred, target_obs, target_mask_obs = predictor(normalized)
    assert y_pred.shape == (1, 2, 1)
    expected = torch.tensor([300.0 - 273.15, 310.0 - 273.15])
    assert torch.allclose(y_pred[0, :, 0], expected)
    assert target_obs is not None and target_obs.shape == (1, 2, 1)
    assert target_mask_obs is not None and target_mask_obs.shape == (1, 2, 1)


def test_normalize_batch_omits_observation_data(
    predictor: NWPInterpolatorPredictor,
) -> None:
    # requires_observation is False -> _unwrap_fields_to_model_batch skips it
    batch = _make_batch(2, [300.0, 310.0])
    normalized = predictor._normalize_batch(batch)
    assert "observation_data" not in normalized


def test_query_locations_not_supported(predictor: NWPInterpolatorPredictor) -> None:
    batch = _make_batch(1, [300.0])
    normalized = predictor._normalize_batch(batch)
    with pytest.raises(NotImplementedError):
        predictor(normalized, query_locations=torch.zeros(1, 2))


def _fake_dataset() -> Any:
    field_config = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
                "geo": {
                    "coords": "stations",
                    "inputs": ["coords.x", "coords.y"],
                    "out": ["x", "y"],
                    "query": True,
                },
                "topo": {"coords": "stations", "inputs": ["topo.a"], "query": True},
            },
            "roles": {
                "target": "obs",
                "global": "nwp",
                "geo_bias": "geo",
                "topo": "topo",
            },
            "transform_type": "real",
        }
    )
    obs_to_nwp = {"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)}
    return types.SimpleNamespace(field_config=field_config, obs_to_nwp=obs_to_nwp)


def test_create_baseline_predictor_builds_from_dataset() -> None:
    built = create_baseline_predictor(_fake_dataset())
    assert isinstance(built, NWPInterpolatorPredictor)
    assert built.output_var_names == ["temperature"]
    assert built.model.nwp_names == ["T_2M"]
    assert built.field_roles == {
        "target": "obs",
        "global": "nwp",
        "geo_bias": "geo",
        "topo": "topo",
    }
