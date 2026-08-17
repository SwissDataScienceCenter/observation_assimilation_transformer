# test/oat_test/predictor_module/test_baseline_predictor.py
import os
import types
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from oat.data_module.field import Field
from oat.data_module.field_spec import FieldConfig, parse_field_config
from oat.model_module.baseline import NWPInterpolatorConfig
from oat.predictor_module.baseline_predictor import (
    NWPInterpolatorPredictor,
    create_baseline_predictor,
)

FIELD_ROLES = {"target": "obs", "global": "nwp", "geo_bias": "geo", "topo": "topo"}


def _make_batch(n: int, t_2m_values: list[float]) -> dict[str, Any]:
    coords = torch.zeros(1, n, 3)
    coords[..., 2] = 100.0  # epoch hours
    nwp_vals = torch.tensor(t_2m_values).reshape(1, n, 1)
    query_fields = {
        "obs": Field(
            coords.clone(), torch.zeros(1, n, 1), ["temperature"], torch.ones(1, n, 1)
        ),
        "nwp": Field(coords.clone(), nwp_vals, ["T_2M"]),
        "geo": Field(coords.clone(), torch.zeros(1, n, 3), ["x", "y", "d"]),
        "topo": Field(coords.clone(), torch.zeros(1, n, 2), ["a", "b"]),
    }
    return {"query_data": {"fields": query_fields}}


@pytest.fixture
def config() -> NWPInterpolatorConfig:
    return NWPInterpolatorConfig(
        nwp_names=["T_2M"],
        obs_names=["temperature"],
        obs_to_nwp={"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)},
    )


@pytest.fixture
def predictor(config: NWPInterpolatorConfig) -> NWPInterpolatorPredictor:
    return NWPInterpolatorPredictor(
        config=config, output_var_names=["temperature"], field_roles=FIELD_ROLES
    )


def test_predict_maps_nwp_to_obs_units(predictor: NWPInterpolatorPredictor) -> None:
    batch = _make_batch(2, [300.0, 310.0])
    normalized = predictor._normalize_batch(batch)
    y_pred, target_obs, target_mask_obs = predictor(normalized)
    assert y_pred.shape == (1, 2, 1)
    expected = torch.tensor([300.0 - 273.15, 310.0 - 273.15])
    assert torch.allclose(y_pred[0, :, 0], expected)
    assert target_obs is not None and target_obs.shape == (1, 2, 1)
    assert target_mask_obs is not None and target_mask_obs.shape == (1, 2, 1)


def test_normalize_batch_omits_observation_data(
    predictor: NWPInterpolatorPredictor,
) -> None:
    # requires_observation is False -> _unwrap_fields_to_model_batch skips it
    batch = _make_batch(2, [300.0, 310.0])
    normalized = predictor._normalize_batch(batch)
    assert "observation_data" not in normalized


def test_query_locations_not_supported(predictor: NWPInterpolatorPredictor) -> None:
    batch = _make_batch(1, [300.0])
    normalized = predictor._normalize_batch(batch)
    with pytest.raises(NotImplementedError):
        predictor(normalized, query_locations=torch.zeros(1, 2))


def _fake_dataset() -> Any:
    field_config = parse_field_config(
        {
            "coord_sources": {"stations": {"from": "obs"}},
            "fields": {
                "obs": {"coords": "stations", "inputs": ["obs.temperature"]},
                "nwp": {"coords": "stations", "inputs": ["nwp.T_2M"], "query": True},
                "geo": {
                    "coords": "stations",
                    "inputs": ["coords.x", "coords.y"],
                    "out": ["x", "y"],
                    "query": True,
                },
                "topo": {"coords": "stations", "inputs": ["topo.a"], "query": True},
            },
            "roles": {
                "target": "obs",
                "global": "nwp",
                "geo_bias": "geo",
                "topo": "topo",
            },
        }
    )
    obs_to_nwp = {"temperature": (["T_2M"], lambda n: n["T_2M"] - 273.15)}
    return types.SimpleNamespace(field_config=field_config, obs_to_nwp=obs_to_nwp)


def test_create_baseline_predictor_builds_from_dataset() -> None:
    built = create_baseline_predictor(_fake_dataset())
    assert isinstance(built, NWPInterpolatorPredictor)
    assert built.output_var_names == ["temperature"]
    assert built.model.nwp_names == ["T_2M"]
    assert built.field_roles == {
        "target": "obs",
        "global": "nwp",
        "geo_bias": "geo",
        "topo": "topo",
    }


def _compose_field_config(dataset_config_name: str) -> FieldConfig:
    """Compose a live dataset config via Hydra and parse its field schema.

    No data I/O: this only parses the YAML-derived dataset config dict, it
    never opens the underlying zarr/parquet dataset files.
    """
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config", overrides=[f"dataset={dataset_config_name}"]
        )
    container = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(container, dict)
    return parse_field_config(container)


@pytest.mark.parametrize(
    "dataset_config_name,obs_to_nwp_module,obs_to_nwp_attr",
    [
        (
            "cosmo_peakweather/temp",
            "oat.data_module.cosmo_peakweather_dataset",
            "COSMO_OBS_TO_NWP",
        ),
        (
            "cosmo_peakweather/temp_precip",
            "oat.data_module.cosmo_peakweather_dataset",
            "COSMO_OBS_TO_NWP",
        ),
        (
            "era5_meteostat/temp",
            "oat.data_module.era5_meteostat_dataset",
            "ERA5_OBS_TO_NWP",
        ),
        (
            "era5_meteostat/temp_precip",
            "oat.data_module.era5_meteostat_dataset",
            "ERA5_OBS_TO_NWP",
        ),
    ],
)
def test_create_baseline_predictor_builds_from_live_dataset_configs(
    dataset_config_name: str, obs_to_nwp_module: str, obs_to_nwp_attr: str
) -> None:
    """Binds the real per-dataset obs_to_nwp table to the real field configs.

    Guards against future drift: someone adding an obs var with no entry in
    the live COSMO/ERA5 obs_to_nwp table, or dropping an NWP input a mapping
    depends on from a field config, should fail here with a named test
    failure instead of surfacing only as a runtime ValueError in a plot/eval
    script (via ``maybe_build_nwp_baseline``/``create_baseline_predictor``).
    """
    import importlib

    obs_to_nwp = getattr(importlib.import_module(obs_to_nwp_module), obs_to_nwp_attr)
    field_config = _compose_field_config(dataset_config_name)
    dataset = types.SimpleNamespace(field_config=field_config, obs_to_nwp=obs_to_nwp)

    built = create_baseline_predictor(dataset)

    assert isinstance(built, NWPInterpolatorPredictor)
    roles = field_config.roles
    expected_nwp_names = list(field_config.fields[roles["global"]].out)
    expected_obs_names = list(field_config.fields[roles["target"]].out)
    assert expected_nwp_names, "global field has no output channels"
    assert expected_obs_names, "target field has no output channels"
    assert built.model.nwp_names == expected_nwp_names
    assert built.output_var_names == expected_obs_names
    assert built.model.obs_names == expected_obs_names
