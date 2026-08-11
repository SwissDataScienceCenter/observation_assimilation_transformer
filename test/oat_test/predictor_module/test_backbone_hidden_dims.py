"""setup_backbone_cfgs builds the Point/MLP encoder hidden stack from the
configurable num_hidden_layers / hidden_width knobs (width defaults to
d_model//2 when unspecified), mirroring the old topographic encoder."""

from omegaconf import OmegaConf

from oat.predictor_module.oat_predictor import setup_backbone_cfgs


def _cfg(d_model: int, ge_extra: dict | None = None, te_extra: dict | None = None):
    ge = {"type": "Point", "act_fn": "GELU", **(ge_extra or {})}
    te = {"type": "MLP", "act_fn": "GELU", "use_layernorm": False, **(te_extra or {})}
    return OmegaConf.create(
        {
            "model": {
                "d_model": d_model,
                "global_encoder": ge,
                "topographic_encoder": te,
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


def test_width_defaults_to_half_d_model_when_unspecified():
    d_model = 32
    cfg = _cfg(
        d_model,
        ge_extra={"num_hidden_layers": 2, "hidden_width": None},
        te_extra={"num_hidden_layers": 2, "hidden_width": None},
    )
    ge, te, _ = setup_backbone_cfgs(cfg, _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [d_model // 2, d_model // 2]
    assert list(te.hidden_dims) == [d_model // 2, d_model // 2]


def test_default_when_knobs_omitted():
    # No num_hidden_layers / hidden_width keys at all -> 2 layers of d_model//2.
    ge, te, _ = setup_backbone_cfgs(_cfg(32), _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [16, 16]
    assert list(te.hidden_dims) == [16, 16]


def test_configurable_width_overrides_default():
    cfg = _cfg(
        32,
        ge_extra={"hidden_width": 64},
        te_extra={"hidden_width": 8},
    )
    ge, te, _ = setup_backbone_cfgs(cfg, _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [64, 64]  # 2 layers (default) of width 64
    assert list(te.hidden_dims) == [8, 8]


def test_configurable_num_layers():
    cfg = _cfg(
        32,
        ge_extra={"num_hidden_layers": 3},
        te_extra={"num_hidden_layers": 0},
    )
    ge, te, _ = setup_backbone_cfgs(cfg, _FIELD_EXTENT)
    assert list(ge.hidden_dims) == [16, 16, 16]  # 3 layers of default width
    assert list(te.hidden_dims) == []  # 0 hidden layers -> single output linear
