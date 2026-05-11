import warnings
import pytest
import torch
from torch.distributions import constraints
from pathlib import Path

from oat.data_module.variable_catalog import (
    VariableDef,
    load_catalog,
    check_distribution_compatibility,
    check_support_overlap,
)
from oat.model_module.distributions import NormalOAT, ExponentialOAT, ZeroInflatedLogNormal


def test_variable_def_accepts_constraint_instances():
    vd = VariableDef(
        name="x",
        description="test",
        unit="units",
        support=constraints.nonnegative,
    )
    assert bool(vd.support.check(torch.tensor(0.0)))
    assert not bool(vd.support.check(torch.tensor(-1.0)))


def test_variable_def_rejects_string_support():
    with pytest.raises(TypeError, match="Constraint"):
        VariableDef(
            name="x",
            description="test",
            unit="units",
            support="nonnegative",  # type: ignore[arg-type]
        )


def test_support_overlap_full_when_subset():
    assert check_support_overlap(constraints.nonnegative, constraints.real) == "full"


def test_support_overlap_partial_when_intersecting_not_subset():
    assert check_support_overlap(constraints.real, constraints.nonnegative) == "partial"


def test_support_overlap_none_when_disjoint():
    assert check_support_overlap(constraints.less_than(0.0), constraints.greater_than(0.0)) == "none"


def test_compatibility_full_overlap_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility(
            "wind_speed",
            constraints.nonnegative,
            "zero_inflated_lognormal",
            ZeroInflatedLogNormal,
        )
        check_distribution_compatibility("temperature", constraints.real, "normal", NormalOAT)


def test_compatibility_none_overlap_raises():
    with pytest.raises(ValueError, match="not fully supported"):
        check_distribution_compatibility(
            "signed_var",
            constraints.less_than(0.0),
            "exponential",
            ExponentialOAT,
        )


def test_compatibility_partial_overlap_warns():
    with pytest.warns(UserWarning, match="partially overlaps"):
        check_distribution_compatibility(
            "temperature",
            constraints.real,
            "exponential",
            ExponentialOAT,
        )


def test_use_log_effective_support_full_overlap_ok():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_distribution_compatibility("some_positive_var", constraints.real, "normal", NormalOAT)


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
    assert bool(catalog["temperature"].support.check(torch.tensor(-1.0)))
    assert catalog["temperature"].default_distribution == "normal"
    assert bool(catalog["wind_speed"].support.check(torch.tensor(0.0)))
    assert not bool(catalog["wind_speed"].support.check(torch.tensor(-1.0)))
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


def test_load_catalog_parses_named_support(tmp_path):
    yaml_content = """
variables:
  wind_speed:
    description: "10m wind speed"
    unit: "m/s"
    support: "nonnegative"
"""
    catalog_file = tmp_path / "catalog_named_support.yaml"
    catalog_file.write_text(yaml_content)

    catalog = load_catalog(catalog_file)
    assert bool(catalog["wind_speed"].support.check(torch.tensor(0.0)))
    assert not bool(catalog["wind_speed"].support.check(torch.tensor(-1.0)))


def test_load_catalog_parses_parameterized_support(tmp_path):
    yaml_content = """
variables:
  humidity:
    description: "relative humidity"
    unit: "fraction"
    support:
      name: "interval"
      lower_bound: 0.0
      upper_bound: 1.0
"""
    catalog_file = tmp_path / "catalog_interval_support.yaml"
    catalog_file.write_text(yaml_content)

    catalog = load_catalog(catalog_file)
    support = catalog["humidity"].support
    assert bool(support.check(torch.tensor(0.2)))
    assert not bool(support.check(torch.tensor(1.2)))


# ──────────────────────────────────────────────
# COSMO-PeakWeather catalog coverage
# ──────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parents[3] / "configs" / "dataset" / "cosmo_peakweather" / "var_catalogue.yaml"


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

from unittest.mock import MagicMock


def _make_mock_dataset(obs_vars, var_settings=None, catalog_path=None, obs_variable_to_distribution=None):
    """Build a minimal mock that exercises get_transform() without real data."""
    from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherDataset

    n_stations = 5
    n_obs_vars = len(obs_vars)
    n_nwp_vars = 3
    n_topo_vars = 2
    # Use varied positive values (1..5) so std != 0 and max != 0.
    obs_tensor = (torch.arange(n_stations, dtype=torch.float32) + 1.0).unsqueeze(1).expand(-1, n_obs_vars)
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
    negative_var:
        description: "a strictly negative variable"
        unit: "units"
        support:
            name: "interval"
            lower_bound: -2.0
            upper_bound: -1.0
        default_distribution: "normal"
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
    from oat.data_module.transform import TranslateScaleTransform
    assert isinstance(output_transform, TranslateScaleTransform)


def test_get_transform_unregistered_dist_name_skips_validation(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["negative_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"negative_var": "unsupported_dist"},
    )
    post_transform, output_transform = ds.get_transform(num_data_points=2)
    assert post_transform is not None
    assert output_transform is not None


def test_get_transform_wider_dist_warns(two_var_catalog):
    ds = _make_mock_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "exponential"},
    )
    with pytest.warns(UserWarning, match="partially overlaps"):
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


# ──────────────────────────────────────────────
# TrainDataset.get_transform() catalog integration
# ──────────────────────────────────────────────

import numpy as np
from oat.data_module.data_split import TrainDataset


def _make_train_dataset(obs_vars, var_settings=None, catalog_path=None, obs_variable_to_distribution=None):
    """Build a minimal mock TrainDataset that exercises get_transform() without real data."""
    n_stations = 5
    n_obs_vars = len(obs_vars)
    n_topo_vars = 2
    n_geo_vars = 3
    n_nwp_vars = 3

    # Use varied positive values (1..5) so std != 0 and max != 0.
    obs_tensor = (torch.arange(n_stations, dtype=torch.float32) + 1.0).unsqueeze(1).expand(-1, n_obs_vars)
    topo_tensor = torch.zeros(n_stations, n_topo_vars, 4, 4)
    geo_tensor = torch.zeros(n_stations, n_geo_vars)
    nwp_field = torch.zeros(n_nwp_vars + n_topo_vars, 8, 8)

    # Sample returned by the underlying dataset (all stations, no obs/query split)
    fake_sample = {
        "global_data": {"global_field": nwp_field},
        "observation_data": {
            "obs": obs_tensor,
            "topo": topo_tensor,
            "geo": geo_tensor,
        },
    }

    inner_ds = MagicMock()
    inner_ds.stations = np.array([f"s{i}" for i in range(n_stations)])
    inner_ds.__len__ = MagicMock(return_value=10)
    inner_ds.__getitem__ = MagicMock(return_value=fake_sample)

    cfg = MagicMock()
    cfg.obs_vars = obs_vars
    cfg.nwp_vars = ["T_2M", "T_G", "DEM"]
    cfg.topo_vars = ["DEM", "TPI_500M"]
    cfg.geo_vars = ["easting", "northing", "dem"]
    cfg.var_settings = var_settings or {}
    cfg.variable_catalog_path = catalog_path
    cfg.obs_variable_to_distribution = obs_variable_to_distribution or {v: "normal" for v in obs_vars}
    inner_ds.config = cfg

    train_indices = np.array([0, 1, 2])
    return TrainDataset(inner_ds, train_indices, split_fraction=1.0)


def test_train_dataset_get_transform_no_catalog_no_error():
    td = _make_train_dataset(obs_vars=["temperature"], catalog_path=None)
    post_transform, output_transform = td.get_transform(num_data_points=2)
    assert post_transform is not None
    assert output_transform is not None


def test_train_dataset_get_transform_no_log_by_default(two_var_catalog):
    from oat.data_module.transform import TranslateScaleTransform
    td = _make_train_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "normal"},
    )
    _post, output_transform = td.get_transform(num_data_points=2)
    assert isinstance(output_transform, TranslateScaleTransform)


def test_train_dataset_get_transform_use_log_positive_var_ok(two_var_catalog):
    from oat.data_module.transform import ComposedTransform, LogTransform
    td = _make_train_dataset(
        obs_vars=["positive_var"],
        var_settings={"positive_var": {"use_log": True}},
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"positive_var": "normal"},
    )
    _post, output_transform = td.get_transform(num_data_points=2)
    assert isinstance(output_transform, ComposedTransform)
    assert any(isinstance(t, LogTransform) for t in output_transform.transforms)


def test_train_dataset_get_transform_use_log_requires_positive_support(two_var_catalog):
    td = _make_train_dataset(
        obs_vars=["real_var"],
        var_settings={"real_var": {"use_log": True}},
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "normal"},
    )
    with pytest.raises(ValueError, match="use_log"):
        td.get_transform(num_data_points=2)


def test_train_dataset_get_transform_unregistered_dist_name_skips_validation(two_var_catalog):
    td = _make_train_dataset(
        obs_vars=["negative_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"negative_var": "unsupported_dist"},
    )
    post_transform, output_transform = td.get_transform(num_data_points=2)
    assert post_transform is not None
    assert output_transform is not None


def test_train_dataset_get_transform_wider_dist_warns(two_var_catalog):
    td = _make_train_dataset(
        obs_vars=["real_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"real_var": "exponential"},
    )
    with pytest.warns(UserWarning, match="partially overlaps"):
        td.get_transform(num_data_points=2)


def test_get_transform_nonneg_max_scale(two_var_catalog):
    """Nonneg variable: translate=0, scale=1/max (no z-score normalisation)."""
    from oat.data_module.transform import TranslateScaleTransform

    # Use a nonneg_var with obs value 2.0 so max=2.0 and expected scale=0.5.
    obs_value = 2.0
    n_stations = 5
    obs_tensor = torch.full((n_stations, 1), obs_value)

    # Build a mock dataset using nonneg_var (support="nonnegative", not log-transformed)
    ds = _make_mock_dataset(
        obs_vars=["nonneg_var"],
        catalog_path=two_var_catalog,
        obs_variable_to_distribution={"nonneg_var": "zero_inflated_lognormal"},
    )
    # Override the fake sample so obs values are obs_value rather than 1.0
    obs_mask = torch.ones(n_stations, 1)
    geo_tensor = torch.zeros(n_stations, 3)
    topo_tensor = torch.zeros(n_stations, 2, 4, 4)
    n_nwp_vars, n_topo_vars = 3, 2
    nwp_field = torch.zeros(n_nwp_vars + n_topo_vars, 8, 8)
    ds.__getitem__ = MagicMock(return_value={
        "global_data": {"global_field": nwp_field},
        "observation_data": {
            "obs": obs_tensor,
            "mask_obs": obs_mask,
            "topo": topo_tensor,
            "geo": geo_tensor,
        },
    })

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _post, output_transform = ds.get_transform(num_data_points=2)

    assert isinstance(output_transform, TranslateScaleTransform)
    # translate must be 0 (no mean subtraction for nonneg variables)
    assert output_transform.translate[0].item() == pytest.approx(0.0)
    # scale must be 1/max = 1/obs_value
    assert output_transform.scale[0].item() == pytest.approx(1.0 / obs_value)
