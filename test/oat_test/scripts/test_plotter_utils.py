"""Tests for scripts/plotter_utils.py path-detection helpers."""

import logging
import types

import torch

from oat.plotting_module import OATPlotter
from oat.plotting_module.legacy import LegacyOATPlotter
from scripts.plotter_utils import (
    is_field_path,
    load_plot_config,
    maybe_build_nwp_baseline,
    select_plotter_cls,
)


def _legacy_dataset():
    return types.SimpleNamespace(
        nwp_extent={"min": torch.zeros(2), "max": torch.ones(2)},
        get_baseline_pre_transform=lambda: (lambda batch: batch),
    )


def _field_dataset():
    return types.SimpleNamespace(
        roles=lambda: {"target": "obs", "global": "nwp"},
        nwp_extent={"min": torch.zeros(2), "max": torch.ones(2)},
    )


def test_is_field_path_true_when_dataset_has_roles():
    assert is_field_path(_field_dataset()) is True


def test_is_field_path_false_when_dataset_has_no_roles():
    assert is_field_path(_legacy_dataset()) is False


def test_select_plotter_cls_field_path():
    assert select_plotter_cls(_field_dataset()) is OATPlotter


def test_select_plotter_cls_legacy_path():
    assert select_plotter_cls(_legacy_dataset()) is LegacyOATPlotter


def test_maybe_build_nwp_baseline_skips_and_warns_on_field_path(caplog):
    with caplog.at_level(logging.WARNING):
        result = maybe_build_nwp_baseline(
            _field_dataset(), ["temperature"], logging.getLogger(__name__)
        )
    assert result is None
    assert "not yet supported" in caplog.text


def test_maybe_build_nwp_baseline_warns_and_returns_none_on_construction_failure(
    caplog,
):
    # NWPInterpolator wraps InterpolationGlobalEncoder, which today
    # unconditionally raises NotImplementedError regardless of data path
    # (a separate, pre-existing issue — see maybe_build_nwp_baseline's
    # docstring). This exercises that real failure path, not a mock.
    with caplog.at_level(logging.WARNING):
        result = maybe_build_nwp_baseline(
            _legacy_dataset(), ["temperature"], logging.getLogger(__name__)
        )
    assert result is None
    assert "unavailable" in caplog.text


def test_load_plot_config_returns_none_for_unrecognized_dataset_type():
    assert load_plot_config(_legacy_dataset()) is None
