"""Shared dataset helpers used by both the COSMO/PeakWeather and ERA5/Meteostat
dataset implementations.
"""

from typing import Any

import torch

from oat.data_module.support import VariableSupport

dtype_map = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_obs_variable_support(catalog: Any, obs_vars: list[str]) -> VariableSupport:
    catalog_obs_vars = catalog.obs_vars
    missing_vars = [var for var in obs_vars if var not in catalog_obs_vars]
    if missing_vars:
        raise KeyError(f"Observation variables missing from catalog: {missing_vars}")

    lower_bounds = torch.tensor(
        [catalog_obs_vars[var].get("lower_bound", float("-inf")) for var in obs_vars],
        dtype=torch.float32,
    )
    upper_bounds = torch.tensor(
        [catalog_obs_vars[var].get("upper_bound", float("inf")) for var in obs_vars],
        dtype=torch.float32,
    )
    lower_open = torch.tensor(
        [catalog_obs_vars[var].get("lower_open", False) for var in obs_vars],
        dtype=torch.bool,
    )
    upper_open = torch.tensor(
        [catalog_obs_vars[var].get("upper_open", False) for var in obs_vars],
        dtype=torch.bool,
    )

    return VariableSupport(
        lower_bounds,
        upper_bounds,
        lower_open=lower_open,
        upper_open=upper_open,
    )
