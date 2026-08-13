"""Shared helpers so diagnostic scripts work against both plotter data paths.

Plotter-class selection and the NWP-baseline gate below key off the
*dataset*, not the predictor: only `DatasetBase` (the Field path) defines
`roles()`, and this signal is accurate even when a script never loads a
real predictor (e.g. evaluate_test_set.py's `--model interpolator` mode).
See docs/superpowers/specs/2026-08-13-oat-plotter-field-migration-design.md.
"""

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
from oat.data_module.legacy.cosmo_peakweather_dataset import (
    CosmoPeakWeatherDataset as _LegacyCosmoPeakWeatherDataset,
)
from oat.data_module.legacy.era5_meteostat_dataset import (
    ERA5MeteostatDataset as _LegacyERA5MeteostatDataset,
)
from oat.plotting_module import OATPlotter
from oat.plotting_module.legacy import LegacyOATPlotter

_PROJECT_ROOT = Path(__file__).parents[1]
_PLOT_CONFIG_PATHS: dict[type, Path] = {
    _ERA5MeteostatDataset: _PROJECT_ROOT
    / "configs/dataset/era5_meteostat/plot_config.yaml",
    _LegacyERA5MeteostatDataset: _PROJECT_ROOT
    / "configs/dataset/era5_meteostat/plot_config.yaml",
    _CosmoPeakWeatherDataset: _PROJECT_ROOT
    / "configs/dataset/cosmo_peakweather/plot_config.yaml",
    _LegacyCosmoPeakWeatherDataset: _PROJECT_ROOT
    / "configs/dataset/cosmo_peakweather/plot_config.yaml",
}


def is_field_path(dataset: Any) -> bool:
    """True if *dataset* is a Field-path dataset (DatasetBase defines roles())."""
    return hasattr(dataset, "roles")


def select_plotter_cls(dataset: Any) -> type:
    """OATPlotter for Field-path datasets, LegacyOATPlotter otherwise."""
    return OATPlotter if is_field_path(dataset) else LegacyOATPlotter


def maybe_build_nwp_baseline(
    dataset: Any,
    output_var_names: list[str],
    logger: logging.Logger,
    *,
    required: bool = False,
) -> Any | None:
    """Build the NWP-interpolator baseline predictor, or None + a warning.

    Returns None (with a warning) if construction fails for any reason.
    Covers two cases: NWPInterpolatorPredictor has no Field-path support
    today (it reads batch['global_data']/batch['query_data'] as flat
    tensors with no field_roles unwrap); and, independently of data path,
    NWPInterpolator wraps InterpolationGlobalEncoder, which today
    unconditionally raises NotImplementedError (the grid-encoder
    deprecation guard added elsewhere on this branch collaterally broke
    this baseline too — confirmed live: `NWPInterpolator(NWPInterpolatorConfig(...))`
    raises NotImplementedError regardless of data path). Reviving that
    guard is out of scope here; this just keeps scripts from crashing on it.

    IMPORTANT (found in final review): OATPlotter._run_inference's
    `predictor if predictor is not None else self.predictor` fallback means
    a bare None returned here is silently interpreted as "use the primary
    model", not "no baseline". Callers whose entire output depends on a
    real baseline (plot_oat_example.py, plot_nwp_domain.py,
    plot_assimilation_panel.py) MUST pass required=True, which raises
    SystemExit(0) instead of returning None — otherwise they'd silently
    render the primary model's own output as if it were the NWP baseline,
    with no error. Only evaluate_test_set.py's --baseline flag (genuinely
    optional, already gated on `if reference is not None`) uses the
    default required=False.
    """
    if is_field_path(dataset):
        logger.warning(
            "NWP baseline comparison is not yet supported on the Field data "
            "path; skipping."
        )
        if required:
            raise SystemExit(0)
        return None
    from oat.model_module.baseline import NWPInterpolatorConfig
    from oat.predictor_module.baseline_predictor import NWPInterpolatorPredictor

    try:
        nwp_config = NWPInterpolatorConfig(field_extent=dataset.nwp_extent)
        return NWPInterpolatorPredictor(
            config=nwp_config,
            output_var_names=output_var_names,
            input_transform=dataset.get_baseline_pre_transform(),
        ).eval()
    except NotImplementedError as exc:
        logger.warning("NWP baseline comparison is unavailable: %s", exc)
        if required:
            raise SystemExit(0) from exc
        return None


def load_plot_config(dataset: Any) -> Any | None:
    """Load obs_plot_config for *dataset*'s type, on either data path."""
    path = _PLOT_CONFIG_PATHS.get(type(dataset))
    if path and path.exists():
        return OmegaConf.load(path).obs_plot_config
    return None
