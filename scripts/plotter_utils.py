"""Shared helpers for the diagnostic and plot scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from oat.data_module.cosmo_peakweather_dataset import (
    CosmoPeakWeatherDataset as _CosmoPeakWeatherDataset,
)
from oat.data_module.era5_meteostat_dataset import (
    ERA5MeteostatDataset as _ERA5MeteostatDataset,
)

_PROJECT_ROOT = Path(__file__).parents[1]
_PLOT_CONFIG_PATHS: dict[type, Path] = {
    _ERA5MeteostatDataset: _PROJECT_ROOT
    / "configs/dataset/era5_meteostat/plot_config.yaml",
    _CosmoPeakWeatherDataset: _PROJECT_ROOT
    / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}


def maybe_build_nwp_baseline(dataset: Any, logger: logging.Logger) -> Any:
    """Build the NWP-interpolator baseline predictor for a field dataset."""
    from oat.predictor_module.baseline_predictor import create_baseline_predictor

    return create_baseline_predictor(dataset).eval()


def load_plot_config(dataset: Any) -> Any | None:
    """Load obs_plot_config for the type of *dataset*."""
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None
