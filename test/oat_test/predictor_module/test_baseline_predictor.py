from typing import Any

import pytest

from oat.model_module.baseline import NWPInterpolatorConfig
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def field_extent() -> dict[str, Any]:
    return {"min": [0.0, 0.0], "max": [100.0, 100.0]}


@pytest.fixture
def nwp_config(field_extent: dict[str, Any]) -> NWPInterpolatorConfig:
    return NWPInterpolatorConfig(field_extent=field_extent)


# ---------------------------------------------------------------------------
# NWPInterpolatorPredictor tests
# ---------------------------------------------------------------------------


def test_nwp_interpolator_predictor_is_deprecated(
    nwp_config: NWPInterpolatorConfig,
) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        NWPInterpolatorPredictor(config=nwp_config, output_var_names=["v0"])
