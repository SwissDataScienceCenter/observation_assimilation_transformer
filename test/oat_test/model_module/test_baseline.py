import os
import pytest
import torch
from typing import Any, Dict

from oat.model_module.baseline import NWPInterpolator, NWPInterpolatorConfig


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


def make_local_data(B: int, S: int) -> Dict[str, torch.Tensor]:
    return {"coords": torch.rand(B, S, 2) * 100.0}


# ---------------------------------------------------------------------------
# NWPInterpolator tests
# ---------------------------------------------------------------------------

def test_nwp_interpolator_output_shape(nwp_config: NWPInterpolatorConfig) -> None:
    B, C, H, W, S = 2, 6, 32, 32, 12
    model = NWPInterpolator(nwp_config)
    global_data = make_global_data(B, C, H, W)
    query_data = make_local_data(B, S)

    out = model(global_data, query_data)

    assert out.shape == (B, S, C)



def test_nwp_interpolator_with_batch_fixture(
    batch_fixture: Dict[str, Any], nwp_config: NWPInterpolatorConfig
) -> None:
    global_data = batch_fixture["global_data"]
    query_data = batch_fixture["query_data"]
    B = global_data["global_field"].shape[0]
    C_global = global_data["global_field"].shape[1]
    S_query = query_data["coords"].shape[1]

    model = NWPInterpolator(nwp_config)
    out = model(global_data, query_data)

    assert out.shape == (B, S_query, C_global)


