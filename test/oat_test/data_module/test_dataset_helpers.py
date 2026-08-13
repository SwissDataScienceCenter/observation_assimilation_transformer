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
