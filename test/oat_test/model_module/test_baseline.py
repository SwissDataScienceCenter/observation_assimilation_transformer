import pytest
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
# NWPInterpolator tests
# ---------------------------------------------------------------------------

def test_nwp_interpolator_is_deprecated(nwp_config: NWPInterpolatorConfig) -> None:
    with pytest.raises(NotImplementedError, match="deprecated"):
        NWPInterpolator(nwp_config)


