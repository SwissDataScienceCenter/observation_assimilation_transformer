"""obs_to_nwp property: surfaces the dataset's NWP->obs conversion table.

Built with object.__new__ (bypassing __init__, which needs real data files):
the property only reads a module-level constant, not any instance state.
"""

from oat.data_module.cosmo_peakweather_dataset import (
    COSMO_OBS_TO_NWP,
    CosmoPeakWeatherDataset,
)
from oat.data_module.era5_meteostat_dataset import (
    ERA5_OBS_TO_NWP,
    ERA5MeteostatDataset,
)


def test_cosmo_obs_to_nwp_returns_module_constant() -> None:
    ds = object.__new__(CosmoPeakWeatherDataset)
    assert ds.obs_to_nwp is COSMO_OBS_TO_NWP


def test_era5_obs_to_nwp_returns_module_constant() -> None:
    ds = object.__new__(ERA5MeteostatDataset)
    assert ds.obs_to_nwp is ERA5_OBS_TO_NWP
