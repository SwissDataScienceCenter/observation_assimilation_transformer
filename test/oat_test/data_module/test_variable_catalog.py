import warnings
import pytest
import torch
from torch.distributions import constraints
from pathlib import Path

from oat.data_module.variable_catalog import (
    VariableDef,
    load_catalog,
    constraint_to_support_str,
    check_distribution_compatibility,
    SUPPORT_RANK,
)
from oat.model_module.distributions import NormalOAT, ExponentialOAT, ZeroInflatedLogNormal, LogNormalOAT


def test_variable_def_valid_support():
    for s in ("real", "nonnegative", "positive"):
        vd = VariableDef(name="x", description="test", unit="units", support=s)
        assert vd.support == s


def test_variable_def_invalid_support():
    with pytest.raises(ValueError, match="Unknown support"):
        VariableDef(name="x", description="test", unit="units", support="complex")


def test_constraint_real():
    assert constraint_to_support_str(constraints.real) == "real"


def test_constraint_nonnegative():
    assert constraint_to_support_str(constraints.nonnegative) == "nonnegative"


def test_constraint_positive():
    assert constraint_to_support_str(constraints.positive) == "positive"


def test_constraint_unknown_returns_none():
    assert constraint_to_support_str(constraints.unit_interval) is None


def test_exact_match_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility("temperature", "real", "normal", NormalOAT)


def test_dist_too_narrow_raises():
    with pytest.raises(ValueError, match="cannot represent all values"):
        check_distribution_compatibility("temperature", "real", "exponential", ExponentialOAT)


def test_dist_wider_than_var_warns():
    with pytest.warns(UserWarning, match="broader support"):
        check_distribution_compatibility("wind_speed", "nonnegative", "normal", NormalOAT)


def test_nonneg_var_nonneg_dist_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility("wind_speed", "nonnegative", "zero_inflated_lognormal", ZeroInflatedLogNormal)


def test_positive_var_nonneg_dist_warns():
    with pytest.warns(UserWarning, match="broader support"):
        check_distribution_compatibility("precip_intensity", "positive", "zero_inflated_lognormal", ZeroInflatedLogNormal)


def test_use_log_effective_support_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility("some_positive_var", "real", "normal", NormalOAT)


def test_use_log_effective_support_wrong_dist_raises():
    with pytest.raises(ValueError, match="cannot represent all values"):
        check_distribution_compatibility("some_positive_var", "real", "exponential", ExponentialOAT)


def test_load_catalog(tmp_path):
    yaml_content = """
variables:
  temperature:
    description: "2m air temperature"
    unit: "C"
    support: "real"
    default_distribution: "normal"
  wind_speed:
    description: "10m wind speed"
    unit: "m/s"
    support: "nonnegative"
    default_distribution: "zero_inflated_lognormal"
  T_2M:
    description: "NWP 2m temperature"
    unit: "K"
    support: "real"
"""
    catalog_file = tmp_path / "catalog.yaml"
    catalog_file.write_text(yaml_content)
    catalog = load_catalog(catalog_file)
    assert "temperature" in catalog
    assert catalog["temperature"].support == "real"
    assert catalog["temperature"].default_distribution == "normal"
    assert catalog["wind_speed"].support == "nonnegative"
    assert catalog["T_2M"].default_distribution is None


def test_load_catalog_invalid_support(tmp_path):
    yaml_content = """
variables:
  bad_var:
    description: "bad"
    unit: "units"
    support: "complex"
"""
    catalog_file = tmp_path / "bad_catalog.yaml"
    catalog_file.write_text(yaml_content)
    with pytest.raises(ValueError, match="Unknown support"):
        load_catalog(catalog_file)
