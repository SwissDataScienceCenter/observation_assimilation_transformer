import torch
import pytest

from oat.model_module.topographic_encoder import MLPTopographyEncoder


def test_mlp_topography_encoder_pointwise():
    enc = MLPTopographyEncoder(
        in_channels=5, out_channels=8, hidden_dims=[16], act_fn="GELU"
    )
    out = enc(torch.randn(2, 7, 5))  # [B, N, C] -> [B, N, out]
    assert out.shape == (2, 7, 8)


def test_mlp_topography_encoder_batchnorm_pointwise():
    enc = MLPTopographyEncoder(
        in_channels=5,
        out_channels=8,
        hidden_dims=[16],
        act_fn="GELU",
        use_batchnorm=True,
    )
    out = enc(torch.randn(2, 7, 5))  # [B, N, C] -> [B, N, out]
    assert out.shape == (2, 7, 8)


def test_mlp_topography_encoder_batchnorm_pools_over_all_leading_dims():
    """BatchNorm1d stats must pool over every leading dim (B*N), not just B."""
    enc = MLPTopographyEncoder(
        in_channels=3, out_channels=4, hidden_dims=[6], use_batchnorm=True
    )
    enc.eval()  # use running stats, deterministic regardless of batch content
    out_batched = enc(torch.randn(2, 5, 3))
    out_single = enc(torch.randn(1, 1, 3))
    assert out_batched.shape == (2, 5, 4)
    assert out_single.shape == (1, 1, 4)


def test_topo_dispatch_mlp():
    from oat.model_module import model as M

    tc = M.TopographicEncoderConfig(
        in_channels=5, out_channels=8, type="MLP", hidden_dims=()
    )

    class _Holder:
        topographic_encoder = tc

    enc = M.UninformedOAT._init_topographic_encoder(
        M.UninformedOAT.__new__(M.UninformedOAT), _Holder
    )
    assert isinstance(enc, MLPTopographyEncoder)


def test_build_topographic_encoder_dispatches_mlp():
    from oat.model_module import model as M

    tc = M.TopographicEncoderConfig(
        in_channels=5, out_channels=8, type="MLP", hidden_dims=()
    )
    enc = M.build_topographic_encoder(tc)
    assert isinstance(enc, MLPTopographyEncoder)


def test_build_topographic_encoder_wires_use_batchnorm():
    from oat.model_module import model as M
    from oat.model_module.topographic_encoder import _FlattenedBatchNorm1d

    tc = M.TopographicEncoderConfig(
        in_channels=5,
        out_channels=8,
        type="MLP",
        hidden_dims=(16,),
        use_batchnorm=True,
    )
    enc = M.build_topographic_encoder(tc)
    assert any(isinstance(m, _FlattenedBatchNorm1d) for m in enc.model)


def test_build_topographic_encoder_rejects_unknown_type():
    from oat.model_module import model as M

    tc = M.TopographicEncoderConfig(in_channels=5, out_channels=8, type="bogus")
    with pytest.raises(ValueError, match="Unsupported topographic encoder type"):
        M.build_topographic_encoder(tc)
