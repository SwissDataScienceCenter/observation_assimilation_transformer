import pytest
import torch
from types import SimpleNamespace

from oat.model_module import global_encoder as ge_mod
from oat.model_module.global_encoder import PseudoTokenGridGlobalEncoder

FIELD_EXTENT = {
    "min": torch.tensor([45.0, 5.0]),
    "max": torch.tensor([49.0, 11.0]),
}


def _make_encoder(
    in_channels: int = 3, out_channels: int = 4
) -> PseudoTokenGridGlobalEncoder:
    return PseudoTokenGridGlobalEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        field_extent=FIELD_EXTENT,
        embed_dim=16,
        num_heads=2,
        n_layers=1,
        grid_shape=(8, 8),
        window_sizes=(4, 4),
        shift_sizes=(2, 2),
        top_k_grid=4,
        top_k_readout=4,
    )


def _inputs(B: int = 2, C: int = 3, H: int = 12, W: int = 12, S: int = 6):
    global_data = {"global_field": torch.randn(B, C, H, W)}
    lat = torch.rand(B, S, 1) * 4.0 + 45.0
    lon = torch.rand(B, S, 1) * 6.0 + 5.0
    local_data = {"coords": torch.cat([lat, lon], dim=-1)}
    return global_data, local_data


def test_output_shape() -> None:
    out = _make_encoder(out_channels=4)(*_inputs(B=2, S=6))
    assert out.shape == (2, 6, 4)


def test_varies_batch_and_station_count() -> None:
    enc = _make_encoder()
    for batch_size in (1, 3):
        for num_stations in (1, 10):
            assert enc(*_inputs(B=batch_size, S=num_stations)).shape[:2] == (
                batch_size,
                num_stations,
            )


def test_readout_permutation_equivariant() -> None:
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    global_data, local_data = _inputs(B=1, S=8)
    out = enc(global_data, local_data)
    perm = torch.randperm(8)
    out_perm = enc(global_data, {"coords": local_data["coords"][:, perm]})
    assert torch.allclose(out[:, perm], out_perm, atol=1e-5)


def test_gradients_flow_to_shared_latent_and_query_token() -> None:
    enc = _make_encoder()
    enc(*_inputs()).pow(2).mean().backward()
    assert enc.grid_latent.grad is not None and enc.grid_latent.grad.abs().sum() > 0
    assert enc.query_token.grad is not None and enc.query_token.grad.abs().sum() > 0


def test_grid_latent_is_shared_no_absolute_embedding() -> None:
    enc = _make_encoder()
    assert enc.grid_latent.shape == (16,)
    for module in enc.modules():
        assert "SphericalHarmonics" not in type(module).__name__
        assert "FourierEmbedding" not in type(module).__name__


def test_window_must_divide_grid() -> None:
    with pytest.raises(ValueError, match="must divide"):
        PseudoTokenGridGlobalEncoder(
            in_channels=3,
            out_channels=4,
            field_extent=FIELD_EXTENT,
            embed_dim=16,
            num_heads=2,
            n_layers=1,
            grid_shape=(8, 8),
            window_sizes=(3, 3),
            shift_sizes=(1, 1),
            top_k_grid=4,
            top_k_readout=4,
        )


def test_encode_decode_matches_forward() -> None:
    """decode(encode_global(...)) must equal a plain forward pass."""
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    global_data, local_data = _inputs(B=2, S=6)

    reference = enc(global_data, local_data)
    encoded = enc.encode_global(global_data)
    split = enc.decode(global_data, local_data, encoded)
    assert torch.allclose(reference, split, atol=1e-5)


def test_shared_encoding_reused_across_readouts() -> None:
    """A single encode_global feeds two different readouts consistently."""
    torch.manual_seed(0)
    enc = _make_encoder().eval()
    global_data, station_data = _inputs(B=2, S=6)
    _, query_data = _inputs(B=2, S=9)

    encoded = enc.encode_global(global_data)
    stations = enc.decode(global_data, station_data, encoded)
    queries = enc.decode(global_data, query_data, encoded)

    assert stations.shape == (2, 6, 4)
    assert queries.shape == (2, 9, 4)
    assert torch.allclose(stations, enc(global_data, station_data), atol=1e-5)
    assert torch.allclose(queries, enc(global_data, query_data), atol=1e-5)


def test_init_dispatch_builds_pseudo_token_encoder() -> None:
    from oat.model_module.model import UninformedOAT

    ge_cfg = SimpleNamespace(
        type="PseudoTokenGrid",
        in_channels=3,
        out_channels=4,
        field_extent_min=(45.0, 5.0),
        field_extent_max=(49.0, 11.0),
        embed_dim=16,
        num_heads=2,
        n_layers=1,
        grid_shape=(8, 8),
        window_sizes=(4, 4),
        shift_sizes=(2, 2),
        top_k_grid=4,
        top_k_readout=4,
        geo_bias_hidden_dim=16,
    )
    cfg = SimpleNamespace(global_encoder=ge_cfg)
    enc = UninformedOAT._init_global_encoder(
        UninformedOAT.__new__(UninformedOAT), cfg
    )
    assert isinstance(enc, ge_mod.PseudoTokenGridGlobalEncoder)
