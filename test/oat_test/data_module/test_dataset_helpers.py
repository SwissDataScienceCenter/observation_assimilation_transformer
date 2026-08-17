from types import SimpleNamespace

import torch

from oat.data_module import dataset_helpers as dh
from oat.data_module.cosmo_peakweather_dataset import (
    DERIVED_FEATURES as COSMO_DERIVED_FEATURES,
)
from oat.data_module.era5_meteostat_dataset import (
    DERIVED_FEATURES as ERA5_DERIVED_FEATURES,
)


def test_dtype_map_has_expected_entries():
    assert dh.dtype_map["float32"] is torch.float32
    assert dh.dtype_map["float64"] is torch.float64


def test_build_obs_variable_support_reads_bounds():
    catalog = SimpleNamespace(obs_vars={"temperature": {"lower_bound": -50.0, "upper_bound": 60.0}})
    vs = dh.build_obs_variable_support(catalog, ["temperature"])
    assert torch.allclose(vs.lower_bounds, torch.tensor([-50.0]))
    assert torch.allclose(vs.upper_bounds, torch.tensor([60.0]))


def test_build_obs_variable_support_missing_var_raises():
    catalog = SimpleNamespace(obs_vars={"temperature": {}})
    try:
        dh.build_obs_variable_support(catalog, ["nope"])
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_derived_features_wind_speed_from_u_v():
    deps, fn = COSMO_DERIVED_FEATURES["wind_speed"]
    assert deps == ["U_10M", "V_10M"]
    u, v = torch.tensor([3.0]), torch.tensor([4.0])
    torch.testing.assert_close(fn(u, v), torch.tensor([5.0]))


def test_derived_features_wind_speed_from_10u_10v():
    deps, fn = ERA5_DERIVED_FEATURES["wind_speed"]
    assert deps == ["10u", "10v"]
    u, v = torch.tensor([3.0]), torch.tensor([4.0])
    torch.testing.assert_close(fn(u, v), torch.tensor([5.0]))
