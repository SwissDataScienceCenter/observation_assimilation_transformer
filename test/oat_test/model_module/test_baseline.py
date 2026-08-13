import pytest
import torch

from oat.model_module.baseline import NWPInterpolator, NWPInterpolatorConfig


def _sub_273(nwp: dict[str, torch.Tensor]) -> torch.Tensor:
    return nwp["T_2M"] - 273.15


def _wind_speed(nwp: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(nwp["U_10M"] ** 2 + nwp["V_10M"] ** 2)


@pytest.fixture
def obs_to_nwp() -> dict:
    return {
        "temperature": (["T_2M"], _sub_273),
        "wind_speed": (["U_10M", "V_10M"], _wind_speed),
    }


@pytest.fixture
def config(obs_to_nwp: dict) -> NWPInterpolatorConfig:
    return NWPInterpolatorConfig(
        nwp_names=["T_2M", "U_10M", "V_10M"],
        obs_names=["temperature", "wind_speed"],
        obs_to_nwp=obs_to_nwp,
    )


def test_requires_observation_is_false() -> None:
    assert NWPInterpolator.requires_observation is False


def test_forward_maps_nwp_to_obs_units(config: NWPInterpolatorConfig) -> None:
    model = NWPInterpolator(config)
    # [1, 3]: T_2M=300K, U_10M=3, V_10M=4
    global_tokens = torch.tensor([[300.0, 3.0, 4.0]])
    out = model(global_tokens)
    assert out.shape == (1, 2)
    assert torch.allclose(out[0, 0], torch.tensor(300.0 - 273.15))
    assert torch.allclose(out[0, 1], torch.tensor(5.0))  # sqrt(3^2 + 4^2)


def test_forward_batched_and_multi_point(config: NWPInterpolatorConfig) -> None:
    model = NWPInterpolator(config)
    global_tokens = torch.rand(2, 5, 3)  # [B, N, C_nwp]
    out = model(global_tokens)
    assert out.shape == (2, 5, 2)


def test_forward_tolerates_extra_trailing_channels(obs_to_nwp: dict) -> None:
    # Extra global channels (e.g. topo.* appended after the NWP scalars) are
    # tolerated: they enter `cols` and are simply unused.
    config = NWPInterpolatorConfig(
        nwp_names=["T_2M", "U_10M", "V_10M", "topo.DEM"],
        obs_names=["temperature"],
        obs_to_nwp=obs_to_nwp,
    )
    model = NWPInterpolator(config)
    global_tokens = torch.tensor([[300.0, 3.0, 4.0, 999.0]])
    out = model(global_tokens)
    assert out.shape == (1, 1)
    assert torch.allclose(out[0, 0], torch.tensor(300.0 - 273.15))


def test_missing_obs_mapping_raises(obs_to_nwp: dict) -> None:
    config = NWPInterpolatorConfig(
        nwp_names=["T_2M"],
        obs_names=["precipitation"],  # not in obs_to_nwp
        obs_to_nwp=obs_to_nwp,
    )
    with pytest.raises(ValueError, match="no NWP mapping"):
        NWPInterpolator(config)


def test_missing_nwp_dependency_raises(obs_to_nwp: dict) -> None:
    config = NWPInterpolatorConfig(
        nwp_names=["T_2M"],  # missing U_10M, V_10M
        obs_names=["wind_speed"],
        obs_to_nwp=obs_to_nwp,
    )
    with pytest.raises(ValueError, match="not in nwp_names"):
        NWPInterpolator(config)
