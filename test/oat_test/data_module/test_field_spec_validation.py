import pytest

from oat.data_module.field_spec import parse_field_config, validate_field_config
from oat.data_module.sources import SourceInfo


def _sources():
    return {
        "obs": SourceInfo(["temperature"], "stations", True, {}),
        "nwp": SourceInfo(["T_2M", "wind_speed"], "nwp_grid", False, {}),
        "topo": SourceInfo(["DEM", "TPI_500M"], None, False, {}),
    }


def _cfg(fields, roles=None):
    return parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}, "nwp_grid": {"from": "nwp"}},
            "fields": fields,
            "roles": roles or {"target": "obs"},
        }
    )


def test_linear_interp_on_masked_source_raises():
    cfg = _cfg(
        {
            "obs": {
                "coords": "stations",
                "inputs": ["obs.temperature"],
                "interp": "linear",
            },
        },
        roles={"target": "obs"},
    )
    with pytest.raises(ValueError, match="mask"):
        validate_field_config(cfg, _sources())


def test_linear_interp_on_unmasked_source_passes():
    cfg = _cfg(
        {
            "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
            "nwp": {
                "coords": "stations",
                "inputs": ["nwp.T_2M"],
                "query": True,
                "interp": "linear",
            },
        },
        roles={"target": "obs"},
    )
    validate_field_config(cfg, _sources())  # no raise


def test_valid_config_passes():
    cfg = _cfg(
        {
            "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
            "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
            "geo": {
                "coords": "stations",
                "inputs": ["coords.x", "coords.y", "topo.DEM"],
                "out": ["e", "n", "dem"],
                "query": True,
            },
        },
        roles={"target": "obs", "geo_bias": "geo"},
    )
    validate_field_config(cfg, _sources())  # no raise


@pytest.mark.parametrize(
    "fields,roles,match",
    [
        (
            {"obs": {"coords": "nowhere", "inputs": ["obs.temperature"]}},
            {"target": "obs"},
            "coord source",
        ),
        (
            {"obs": {"coords": "stations", "inputs": ["nwp.NOPE"]}},
            {"target": "obs"},
            "var",
        ),
        (
            {
                "obs": {
                    "coords": "stations",
                    "inputs": ["obs.temperature"],
                    "out": ["a", "b"],
                }
            },
            {"target": "obs"},
            "out",
        ),
        (
            {
                "g": {
                    "coords": "stations",
                    "inputs": ["nwp.T_2M", "topo.DEM"],
                    "out": ["x", "x"],
                }
            },
            {"target": "g"},
            "unique",
        ),
        (
            {"mix": {"coords": "stations", "inputs": ["obs.temperature", "nwp.T_2M"]}},
            {"target": "mix"},
            "mix",
        ),
        (
            {"obs": {"coords": "stations", "inputs": ["obs.temperature"]}},
            {"target": "missing"},
            "declared",
        ),
        (
            {"nwp": {"coords": "stations", "inputs": ["nwp.T_2M"]}},
            {"target": "nwp"},
            "obs",
        ),
    ],
)
def test_invalid_configs_raise(fields, roles, match):
    cfg = _cfg(fields, roles)
    with pytest.raises(ValueError, match=match):
        validate_field_config(cfg, _sources())
