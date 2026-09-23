import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from scripts.evaluate_test_set import _ckpt_tag, _exclude_context, _print_summary


def test_exclude_context_removes_context_stations():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({"test_context_station_indices": ["b", "d"]})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 2]


def test_exclude_context_noop_without_context_field():
    stations = np.array(["a", "b", "c", "d"])
    cfg_split = OmegaConf.create({})
    result = _exclude_context([0, 1, 2, 3], cfg_split, stations)
    assert sorted(result) == [0, 1, 2, 3]


def test_ckpt_tag_differs_with_and_without_exclude_context():
    """The output filename must differ between the two eval arms.

    Without this, a run with --exclude-context overwrites the CSV and
    summary from a run without it (and vice versa).
    """
    tag_with_context = _ckpt_tag("my_ckpt", exclude_context=False)
    tag_without_context = _ckpt_tag("my_ckpt", exclude_context=True)
    assert tag_with_context != tag_without_context
    assert tag_with_context == "my_ckpt"
    assert tag_without_context == "my_ckpt_nocontext"


def _base_metrics() -> dict:
    return {
        "crps": torch.zeros(2, 1),
        "mae": torch.zeros(2, 1),
        "crps_global": torch.tensor(0.0),
        "mae_global": torch.tensor(0.0),
    }


def test_print_summary_warns_on_nonfinite(caplog):
    """A non-finite prediction must surface as a loud warning, not silently
    vanish into the averaged CRPS/MAE numbers."""
    metrics = {**_base_metrics(), "n_nonfinite": torch.tensor([[1.0], [0.0]])}
    logger = logging.getLogger("test_print_summary_warns")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(metrics, ["prcp"], logger)
    assert "non-finite" in caplog.text


def test_print_summary_silent_when_all_finite(caplog):
    metrics = {**_base_metrics(), "n_nonfinite": torch.zeros(2, 1)}
    logger = logging.getLogger("test_print_summary_silent")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(metrics, ["prcp"], logger)
    assert "non-finite" not in caplog.text


def test_print_summary_handles_missing_n_nonfinite(caplog):
    """Callers that skip compute_station_metrics (the long-dump aggregation
    path) don't produce n_nonfinite at all; must not crash or warn."""
    logger = logging.getLogger("test_print_summary_missing")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _print_summary(_base_metrics(), ["prcp"], logger)
    assert "non-finite" not in caplog.text
