import torch

from oat.model_module.topographic_encoder import MLPTopographyEncoder


def test_mlp_topography_encoder_pointwise():
    enc = MLPTopographyEncoder(
        in_channels=5, out_channels=8, hidden_dims=[16], act_fn="GELU"
    )
    out = enc(torch.randn(2, 7, 5))  # [B, N, C] -> [B, N, out]
    assert out.shape == (2, 7, 8)


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
