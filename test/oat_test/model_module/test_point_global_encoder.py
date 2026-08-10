import torch

from oat.model_module.global_encoder import PointGlobalEncoder


def test_point_global_encoder_shapes() -> None:
    enc = PointGlobalEncoder(
        mlp_params={
            "in_channels": 3,
            "out_channels": 8,
            "hidden_dims": [16],
            "act_fn": "GELU",
        }
    )
    local = {"global_tokens": torch.randn(2, 5, 3), "coords": torch.randn(2, 5, 2)}
    out = enc({}, local)
    assert out.shape == (2, 5, 8)


def test_point_global_encoder_dispatch() -> None:
    from oat.model_module import model as M

    cfg = M.GlobalEncoderConfig(
        in_channels=3,
        field_extent_min=(0.0, 0.0),
        field_extent_max=(1.0, 1.0),
        type="Point",
        out_channels=8,
        hidden_dims=(),
        act_fn="GELU",
    )

    class _Holder:
        global_encoder = cfg

    enc = M.UninformedOAT._init_global_encoder(
        M.UninformedOAT.__new__(M.UninformedOAT), _Holder
    )
    assert isinstance(enc, PointGlobalEncoder)
