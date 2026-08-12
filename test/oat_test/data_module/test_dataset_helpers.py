import torch
from types import SimpleNamespace
from oat.data_module import dataset_helpers as dh


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


def test_derived_features_are_dataset_specific():
    assert dh.COSMO_DERIVED_FEATURES["wind_speed"][0] == ["U_10M", "V_10M"]
    assert dh.ERA5_DERIVED_FEATURES["wind_speed"][0] == ["10u", "10v"]
