"""setup_backbone_cfgs derives 2 hidden layers of d_model//2 for the point-wise
Point/MLP encoders, mirroring the old topographic encoder's structure."""

from omegaconf import OmegaConf

from oat.predictor_module.oat_predictor import setup_backbone_cfgs


def _cfg(d_model: int, ge_type: str, te_type: str):
    return OmegaConf.create(
        {
            "model": {
                "d_model": d_model,
                "global_encoder": {"type": ge_type, "act_fn": "GELU"},
                "topographic_encoder": {
                    "type": te_type,
                    "act_fn": "GELU",
                    "use_layernorm": False,
                },
                "nwp_topo_fusion": {},
            },
            "dataset": {
                "nwp_vars": ["a", "b"],
                "topo_vars": ["t1", "t2", "t3"],
                "use_time_embedding": True,
            },
        }
    )


_FIELD_EXTENT = {"min": [0.0, 0.0], "max": [1.0, 1.0]}


def test_point_and_mlp_encoders_get_two_hidden_layers_of_half_d_model():
    d_model = 32
    ge, te, _ = setup_backbone_cfgs(_cfg(d_model, "Point", "MLP"), _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [d_model // 2, d_model // 2]
    assert list(te.hidden_dims) == [d_model // 2, d_model // 2]
    assert ge.out_channels == d_model // 2
    assert te.out_channels == d_model // 2
    # Point global input is nwp-only (topo arrives via the topo field).
    assert ge.in_channels == 2


def test_derivation_tracks_d_model():
    ge, te, _ = setup_backbone_cfgs(_cfg(64, "Point", "MLP"), _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [32, 32]
    assert list(te.hidden_dims) == [32, 32]
