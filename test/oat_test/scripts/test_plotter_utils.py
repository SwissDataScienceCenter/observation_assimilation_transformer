"""Tests for scripts/plotter_utils.py path-detection helpers."""

import logging
import types

import pytest
import torch

from oat.data_module.field_spec import parse_field_config
from oat.plotting_module import OATPlotter
from oat.plotting_module.legacy import LegacyOATPlotter
from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor
from scripts.plotter_utils import (
    is_field_path,
    load_plot_config,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)


def _legacy_dataset():
    return types.SimpleNamespace(
        nwp_extent={"min": torch.zeros(2), "max": torch.ones(2)},
    )


def _field_dataset():
    return types.SimpleNamespace(
        roles=lambda: {"target": "obs", "global": "nwp"},
        nwp_extent={"min": torch.zeros(2), "max": torch.ones(2)},
    )


def _field_dataset_with_baseline_support():
    field_config = parse_field_config(
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
    return types.SimpleNamespace(
        roles=lambda: {"target": "obs", "global": "nwp"},
        field_config=field_config,
        obs_to_nwp={"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)},
    )


def test_is_field_path_true_when_dataset_has_roles():
    assert is_field_path(_field_dataset()) is True


def test_is_field_path_false_when_dataset_has_no_roles():
    assert is_field_path(_legacy_dataset()) is False


def test_select_plotter_cls_field_path():
    assert select_plotter_cls(_field_dataset()) is OATPlotter


def test_select_plotter_cls_legacy_path():
    assert select_plotter_cls(_legacy_dataset()) is LegacyOATPlotter


def test_maybe_build_nwp_baseline_builds_predictor_on_field_path():
    result = maybe_build_nwp_baseline(
        _field_dataset_with_baseline_support(), logging.getLogger(__name__)
    )
    assert isinstance(result, NWPInterpolatorPredictor)


def test_maybe_build_nwp_baseline_warns_and_returns_none_on_legacy_path(caplog):
    with caplog.at_level(logging.WARNING):
        result = maybe_build_nwp_baseline(
            _legacy_dataset(), logging.getLogger(__name__)
        )
    assert result is None
    assert "not supported" in caplog.text


def test_maybe_build_nwp_baseline_required_raises_systemexit_on_legacy_path(caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(SystemExit):
            maybe_build_nwp_baseline(
                _legacy_dataset(), logging.getLogger(__name__), required=True
            )


def test_load_plot_config_returns_none_for_unrecognized_dataset_type():
    assert load_plot_config(_legacy_dataset()) is None
