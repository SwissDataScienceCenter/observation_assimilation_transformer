import pytest
import torch
from typing import Any, Dict

from oat.model_module.baseline import NWPInterpolatorConfig
from oat.predictor_module.base_predictor import BasePredictor
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def field_extent() -> Dict[str, Any]:
    return {"min": [0.0, 0.0], "max": [100.0, 100.0]}


@pytest.fixture
def nwp_config(field_extent: Dict[str, Any]) -> NWPInterpolatorConfig:
    return NWPInterpolatorConfig(field_extent=field_extent)


# ---------------------------------------------------------------------------
# Synthetic batch helpers
# ---------------------------------------------------------------------------

def make_global_data(B: int, C: int, H: int = 32, W: int = 32) -> Dict[str, torch.Tensor]:
    return {"global_field": torch.randn(B, C, H, W)}


def make_batch(
    B: int,
    C: int,
    S_query: int,
    D_obs: int | None = None,
    include_obs: bool = True,
) -> Dict[str, Any]:
    global_data = make_global_data(B, C)
    query_data: Dict[str, torch.Tensor] = {"coords": torch.rand(B, S_query, 2) * 100.0}
    if include_obs and D_obs is not None:
        query_data["obs"] = torch.randn(B, S_query, D_obs)
        query_data["mask_obs"] = torch.ones(B, S_query, D_obs, dtype=torch.bool)
    return {"global_data": global_data, "query_data": query_data}


# ---------------------------------------------------------------------------
# NWPInterpolatorPredictor tests
# ---------------------------------------------------------------------------

def test_predictor_forward_output_shape(nwp_config: NWPInterpolatorConfig) -> None:
    B, C, S_query, D_obs = 2, 4, 8, 2
    obs_vars = [f"var_{i}" for i in range(C)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    batch = make_batch(B, C, S_query, D_obs=D_obs, include_obs=True)

    predictions, target_obs, target_mask_obs = predictor(batch)

    assert predictions.shape == (B, S_query, C)
    assert target_obs is not None and target_obs.shape == (B, S_query, D_obs)
    assert target_mask_obs is not None and target_mask_obs.shape == (B, S_query, D_obs)


def test_predictor_forward_no_obs_in_query(nwp_config: NWPInterpolatorConfig) -> None:
    B, C, S_query = 2, 3, 6
    obs_vars = [f"var_{i}" for i in range(C)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    batch = make_batch(B, C, S_query, include_obs=False)

    predictions, target_obs, target_mask_obs = predictor(batch)

    assert predictions.shape == (B, S_query, C)
    assert target_obs is None
    assert target_mask_obs is None


def test_predictor_with_batch_fixture(
    batch_fixture: Dict[str, Any], nwp_config: NWPInterpolatorConfig
) -> None:
    global_field = batch_fixture["global_data"]["global_field"]
    query_coords = batch_fixture["query_data"]["coords"]
    B = global_field.shape[0]
    C_global = global_field.shape[1]
    S_query = query_coords.shape[1]

    obs_vars = [f"var_{i}" for i in range(C_global)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    predictions, _, _ = predictor(batch_fixture)

    assert predictions.shape == (B, S_query, C_global)


def test_nwp_predictor_is_base_predictor(nwp_config: NWPInterpolatorConfig) -> None:
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=["v0"])
    assert isinstance(predictor, BasePredictor)


def test_normalize_batch_is_identity(nwp_config: NWPInterpolatorConfig) -> None:
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=["v0"])
    batch = {"global_data": {"global_field": torch.randn(1, 1, 8, 8)}}
    result = predictor._normalize_batch(batch)
    assert result is batch


def test_predict_step_samples_shape(nwp_config: NWPInterpolatorConfig) -> None:
    B, C, S_query, n_samples = 2, 3, 6, 10
    obs_vars = [f"var_{i}" for i in range(C)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    batch = make_batch(B, C, S_query, include_obs=False)
    samples = predictor.predict_step(batch, batch_idx=0, n_samples=n_samples)
    assert samples.shape == (B, S_query, n_samples, C)


def test_predict_step_samples_deterministic(nwp_config: NWPInterpolatorConfig) -> None:
    """All n_samples slices are identical since NWP interpolation is deterministic."""
    B, C, S_query, n_samples = 1, 2, 4, 5
    obs_vars = [f"var_{i}" for i in range(C)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    batch = make_batch(B, C, S_query, include_obs=False)
    samples = predictor.predict_step(batch, batch_idx=0, n_samples=n_samples)
    assert torch.allclose(samples[:, :, 0, :], samples[:, :, 1, :])


def test_predict_step_default_n_samples(nwp_config: NWPInterpolatorConfig) -> None:
    """With no n_samples, returns shape [B, S, 1, C]."""
    B, C, S_query = 1, 2, 4
    obs_vars = [f"var_{i}" for i in range(C)]
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=obs_vars)
    batch = make_batch(B, C, S_query, include_obs=False)
    samples = predictor.predict_step(batch, batch_idx=0)
    assert samples.shape == (B, S_query, 1, C)


def test_predict_step_invalid_return_as(nwp_config: NWPInterpolatorConfig) -> None:
    predictor = NWPInterpolatorPredictor(config=nwp_config, output_var_names=["v0"])
    batch = make_batch(1, 1, 2, include_obs=False)
    with pytest.raises(ValueError):
        predictor.predict_step(batch, batch_idx=0, return_as="distribution")
