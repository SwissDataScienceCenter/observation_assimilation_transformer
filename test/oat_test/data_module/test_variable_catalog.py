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


def test_real_support_with_real_dist_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility("some_positive_var", "real", "normal", NormalOAT)


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


# ──────────────────────────────────────────────
# COSMO-PeakWeather catalog coverage
# ──────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parents[3] / "configs" / "variables" / "cosmo_peakweather.yaml"


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="catalog file not found")
def test_cosmo_peakweather_catalog_loads():
    catalog = load_catalog(CATALOG_PATH)
    obs_vars_with_dist = [name for name, vd in catalog.items() if vd.default_distribution is not None]
    assert "temperature" in obs_vars_with_dist
    assert "wind_gust" in obs_vars_with_dist
    assert catalog["T_2M"].default_distribution is None


@pytest.mark.skipif(not CATALOG_PATH.exists(), reason="catalog file not found")
def test_cosmo_peakweather_catalog_default_distributions_are_valid():
    from oat.model_module.output_head import OutputHead
    catalog = load_catalog(CATALOG_PATH)
    supported = set(OutputHead.supported_distributions())
    for name, vd in catalog.items():
        if vd.default_distribution is not None:
            assert vd.default_distribution in supported, (
                f"Variable '{name}' default_distribution '{vd.default_distribution}' "
                f"not in supported distributions: {supported}"
            )


def test_cosmo_peak_weather_config_accepts_new_fields():
    """CosmoPeakWeatherConfig dataclass accepts variable_catalog_path and var_settings."""
    from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherConfig
    cfg = CosmoPeakWeatherConfig(
        nwp_path="/tmp/nwp",
        topo_path="/tmp/topo",
        obs_path="/tmp/obs",
        ref_nwp_path="/tmp/ref",
        nwp_vars=["T_2M"],
        topo_vars=["DEM"],
        obs_vars=["temperature"],
        variable_catalog_path="/tmp/catalog.yaml",
        var_settings={"temperature": {"use_log": False}},
    )
    assert cfg.variable_catalog_path == "/tmp/catalog.yaml"
    assert cfg.var_settings == {"temperature": {"use_log": False}}


# ──────────────────────────────────────────────
# get_transform() integration tests (no real data)
# ──────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import tempfile


def _make_mock_dataset(obs_vars, var_settings=None, catalog_path=None, obs_variable_to_distribution=None):
    """Build a minimal mock that exercises get_transform() without real data."""
    from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset

    n_stations = 5
    n_obs_vars = len(obs_vars)
    n_nwp_vars = 3
    n_topo_vars = 2
    obs_tensor = torch.ones(n_stations, n_obs_vars)
    obs_mask = torch.ones(n_stations, n_obs_vars)
    geo_tensor = torch.zeros(n_stations, 3)
    topo_tensor = torch.zeros(n_stations, n_topo_vars, 4, 4)
    nwp_field = torch.zeros(n_nwp_vars + n_topo_vars, 8, 8)

    fake_sample = {
        "global_data": {"global_field": nwp_field},
        "observation_data": {
            "obs": obs_tensor,
            "mask_obs": obs_mask,
            "topo": topo_tensor,
            "geo": geo_tensor,
        },
    }

    ds = MagicMock(spec=CosmoPeakWeatherDataset)
    ds.__len__ = MagicMock(return_value=10)
    ds.__getitem__ = MagicMock(return_value=fake_sample)
    cfg = MagicMock()
    cfg.obs_vars = obs_vars
    cfg.nwp_vars = ["T_2M", "T_G", "DEM"]
    cfg.topo_vars = ["DEM", "TPI_500M"]
    cfg.var_settings = var_settings or {}
    cfg.variable_catalog_path = catalog_path
    cfg.obs_variable_to_distribution = obs_variable_to_distribution or {v: "normal" for v in obs_vars}
    ds.config = cfg
    ds.station_geo = geo_tensor
    ds.get_transform = lambda *a, **kw: CosmoPeakWeatherDataset.get_transform(ds, *a, **kw)
    return ds


@pytest.fixture
def two_var_catalog(tmp_path):
    yaml_content = """
variables:
  positive_var:
    description: "a strictly positive variable"
    unit: "units"
    support: "positive"
    default_distribution: "normal"
  real_var:
    description: "a real-valued variable"
    unit: "units"
    support: "real"
    default_distribution: "normal"
  nonneg_var:
    description: "a nonneg variable"
    unit: "units"
    support: "nonnegative"
    default_distribution: "zero_inflated_lognormal"
"""
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml_content)
    return str(p)


def test_get_transform_use_log_requires_positive_support(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["real_var"],
        var_settings={"real_var": {"use_log": True}},
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "normal"},
    )
    with pytest.raises(ValueError, match="use_log"):
        ds.get_transform(num_data_points=2)


def test_get_transform_use_log_positive_var_ok(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["positive_var"],
        var_settings={"positive_var": {"use_log": True}},
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"positive_var": "normal"},
    )
    post_transform, output_transform = ds.get_transform(num_data_points=2)
    from oat.data_module.transform import ComposedTransform, LogTransform
    assert isinstance(output_transform, ComposedTransform)
    assert any(isinstance(t, LogTransform) for t in output_transform.transforms)


def test_get_transform_no_log_by_default(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "normal"},
    )
    post_transform, output_transform = ds.get_transform(num_data_points=2)
    from oat.data_module.transform import LogTransform, StandardNormalizeTransform
    assert isinstance(output_transform, StandardNormalizeTransform)


def test_get_transform_incompatible_dist_raises(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "exponential"},
    )
    with pytest.raises(ValueError, match="cannot represent all values"):
        ds.get_transform(num_data_points=2)


def test_get_transform_wider_dist_warns(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["nonneg_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"nonneg_var": "normal"},
    )
    with pytest.warns(UserWarning, match="broader support"):
        ds.get_transform(num_data_points=2)


def test_get_transform_no_catalog_no_error():
    """Without a catalog, get_transform() should work with no validation."""
    ds = _make_mock_dataset(
        obs_vars=["temperature"],
        catalog_path=None,
        obs_variable_to_distribution={"temperature": "normal"},
    )
    post_transform, output_transform = ds.get_transform(num_data_points=2)
    assert post_transform is not None
    assert output_transform is not None


def test_get_transform_returns_correct_types(two_var_catalog):
    from oat.data_module.transform import NestedTransform, Transform
    ds = _make_mock_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "normal"},
    )
    post_transform, output_transform = ds.get_transform(num_data_points=2)
    assert isinstance(post_transform, NestedTransform)
    assert isinstance(output_transform, Transform)


def test_get_transform_use_log_requires_catalog_entry(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["unknown_var"],
        var_settings={"unknown_var": {"use_log": True}},
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"unknown_var": "normal"},
    )
    with pytest.raises(ValueError, match="not found"):
        ds.get_transform(num_data_points=2)
