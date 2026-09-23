"""COSMO-E1 NWP source combined with Meteostat observations.

``ERA5MeteostatDataset`` with the NWP source swapped from ERA5 to
COSMO-E1. Obs, topo, station coordinates, and NWP sampling machinery are
unchanged: Meteostat's station table stores lon/lat, which already
matches COSMO's native anemoi lon/lat (unlike PeakWeather, which stores
Swiss LV95 coordinates and needs the pyproj reprojection in
``cosmo_peakweather_dataset.py``). Only the NWP variable set and the
obs<->NWP unit mapping differ, so those are the only two overrides here.

See docs/superpowers/specs/2026-09-23-cosmo-meteostat-interpolator-design.md
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from oat.data_module.era5_meteostat_dataset import ERA5MeteostatDataset

# (deps, fn) where fn(*dep_tensors) -> tensor (H, W). Wind speed/direction
# derived from the COSMO U/V 10m wind components. Copied from
# cosmo_peakweather_dataset.DERIVED_FEATURES.
COSMO_METEOSTAT_DERIVED_FEATURES: dict[
    str, tuple[list[str], Callable[..., torch.Tensor]]
] = {
    "wind_speed": (["U_10M", "V_10M"], lambda u, v: torch.sqrt(u**2 + v**2)),
    "wind_direction": (
        ["U_10M", "V_10M"],
        lambda u, v: (torch.rad2deg(torch.atan2(u, v)) + 180.0) % 360.0,
    ),
}

# Mapping from Meteostat obs variable name -> (required COSMO NWP vars,
# conversion fn). wspd/wpgt both fall back to wind *speed* (COSMO has no
# gust field), matching ERA5_OBS_TO_NWP's existing wpgt -> wind-speed
# approximation for the same reason. The precipitation factor (600.0)
# matches cosmo_peakweather_dataset.COSMO_OBS_TO_NWP.
COSMO_METEOSTAT_OBS_TO_NWP: dict[
    str, tuple[list[str], Callable[[dict[str, Tensor]], Tensor]]
] = {
    "temp": (["T_2M"], lambda n: n["T_2M"] - 273.15),
    "wspd": (
        ["U_10M", "V_10M"],
        lambda n: torch.sqrt(n["U_10M"] ** 2 + n["V_10M"] ** 2),
    ),
    "wpgt": (
        ["U_10M", "V_10M"],
        lambda n: torch.sqrt(n["U_10M"] ** 2 + n["V_10M"] ** 2),
    ),
    "prcp": (["TOT_PREC_1H"], lambda n: 600.0 * n["TOT_PREC_1H"]),
}


class CosmoMeteostatDataset(ERA5MeteostatDataset):
    """``ERA5MeteostatDataset`` with COSMO-E1 as the NWP source.

    Set ``config.nwp_path``/``config.nwp_vars`` to the COSMO-E1 zarr and
    its variable names (see ``configs/dataset/cosmo_meteostat/``); every
    other config field (obs, topo, coord_vars) stays the Meteostat one.
    """

    _derived_features = COSMO_METEOSTAT_DERIVED_FEATURES
    _obs_to_nwp_map = COSMO_METEOSTAT_OBS_TO_NWP
