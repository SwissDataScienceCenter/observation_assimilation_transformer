"""Tests for scripts/plotter_utils.py helpers."""

import logging
import types

from oat.data_module.field_spec import parse_field_config
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from scripts.plotter_utils import load_plot_config, maybe_build_nwp_baseline


def _field_dataset_with_baseline_support():
    field_config = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
            },
            "roles": {"target": "obs", "global": "nwp"},
        }
    )
    return types.SimpleNamespace(
        roles=lambda: {"target": "obs", "global": "nwp"},
        field_config=field_config,
        obs_to_nwp={"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)},
    )


def test_maybe_build_nwp_baseline_builds_predictor_on_field_path():
    result = maybe_build_nwp_baseline(
        _field_dataset_with_baseline_support(), logging.getLogger(__name__)
    )
    assert isinstance(result, NWPInterpolatorPredictor)


def test_load_plot_config_returns_none_for_unrecognized_dataset_type():
    assert load_plot_config(types.SimpleNamespace()) is None


def test_plotting_module_exports_only_oat_plotter():
    import oat.plotting_module as pm

    assert pm.__all__ == ["OATPlotter"]
