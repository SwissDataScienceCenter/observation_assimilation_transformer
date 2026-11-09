"""Tests for the optional query-noise sample head in OAT."""

import pytest
import torch
from oat_test.conftest import FIELD_ROLES

from oat.model_module.model import OAT, QueryNoiseConfig, _search_coords
from oat.predictor_module import OATPredictor


def _qn_cfg(**kw) -> QueryNoiseConfig:
    base = dict(
        length_scales_km=(2.0, 8.0),
        km_per_unit=1.0,
        channels_per_scale=2,
        num_frequencies=16,
        mlp_hidden_layers=1,
        train_members=4,
    )
    return QueryNoiseConfig(**{**base, **kw})


def _inputs(cfg, batch_fixture, transform_fixture):
    input_transform, output_transform = transform_fixture
    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(cfg.output_head.variable_to_distribution.keys()),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )
    batch = pred._normalize_batch(batch_fixture)
    query = batch["query_data"]
    target = query.pop("obs")
    mask = query.pop("mask_obs")
    return pred.model, batch, target, mask


def _noise_on(cfg, **kw):
    cfg.query_noise = _qn_cfg(**kw)
    cfg.pre_oat_loss_weight = 0.1
    return cfg


def test_noise_off_has_no_new_modules_or_keys(model_cfg_fixture):
    model = OAT(model_cfg_fixture)
    assert not hasattr(model, "noise_field")
    assert not hasattr(model, "sample_head")
    for key in model.state_dict():
        assert "noise_field" not in key and "sample_head" not in key


def test_noise_on_forward_samples_shape(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(model_cfg_fixture)
    model, batch, _, _ = _inputs(cfg, batch_fixture, transform_fixture)
    model.eval()
    out = model.forward(
        batch["global_data"],
        batch["observation_data"],
        batch["query_data"],
        return_pre_oat_output=True,
    )
    assert out["samples"].shape == (2, 5, cfg.num_samples, 2)
    assert "pre_oat_output" in out


def test_noise_on_loss_finite_and_gradients_reach_head_and_decoder(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(model_cfg_fixture)
    model, batch, target, mask = _inputs(cfg, batch_fixture, transform_fixture)
    out, _, loss = model.forward_with_loss(
        batch["global_data"],
        batch["observation_data"],
        batch["query_data"],
        target,
        mask,
        return_pre_oat_output=True,
    )
    assert torch.isfinite(loss)
    assert "pre_oat_output" in out
    assert out["samples"].shape == (2, 5, cfg.num_samples, 2)
    loss.backward()
    for module in (model.sample_head, model.oat_block):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert any(g.abs().sum() > 0 for g in grads)


def test_same_noise_draw_same_samples(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(model_cfg_fixture)
    model, batch, _, _ = _inputs(cfg, batch_fixture, transform_fixture)
    model.eval()
    coord_dim = _search_coords(batch["query_data"]).shape[-1]
    draw = model.noise_field.draw(2, 6, coord_dim, torch.device("cpu"), torch.float32)
    args = (batch["global_data"], batch["observation_data"], batch["query_data"])
    with torch.no_grad():
        a = model.forward(*args, noise_draw=draw)["samples"]
        b = model.forward(*args, noise_draw=draw)["samples"]
    assert a.shape[2] == 6
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", ["members", "innovation_without_pre_oat"])
def test_noise_config_validation(model_cfg_fixture, bad):
    cfg = model_cfg_fixture
    if bad == "members":
        cfg.query_noise = _qn_cfg(train_members=1)
        cfg.pre_oat_loss_weight = 0.1
    else:
        cfg.query_noise = _qn_cfg()
        cfg.pre_oat_loss_weight = 0.0
    assert cfg.output_head.observation_assimilation_as == "innovation"
    with pytest.raises(ValueError):
        OAT(cfg)


def _loss(model, batch, target, mask):
    return model.forward_with_loss(
        batch["global_data"],
        batch["observation_data"],
        batch["query_data"],
        target,
        mask,
    )


def test_vs_weight_zero_gives_same_loss_and_state_dict(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(model_cfg_fixture)
    torch.manual_seed(0)
    m1, batch, target, mask = _inputs(cfg, batch_fixture, transform_fixture)
    cfg0 = _noise_on(model_cfg_fixture, vs_weight=0.0)
    torch.manual_seed(0)
    m2, _, _, _ = _inputs(cfg0, batch_fixture, transform_fixture)
    cfg5 = _noise_on(model_cfg_fixture, vs_weight=0.5)
    m5, _, _, _ = _inputs(cfg5, batch_fixture, transform_fixture)
    for m in (m1, m2):
        m.eval()
    assert list(m1.state_dict()) == list(m2.state_dict()) == list(m5.state_dict())
    m2.load_state_dict(m1.state_dict())
    members = torch.randn(2, 5, 4, 2)
    for m in (m1, m2):
        m._noise_samples = lambda *a, **k: members
    out1, _, l1 = _loss(m1, batch, target, mask)
    out2, _, l2 = _loss(m2, batch, target, mask)
    assert torch.allclose(l1, l2)
    assert "loss_vs" not in out1 and "loss_vs" not in out2


def test_vs_term_is_inside_oat_part_of_blend(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(model_cfg_fixture, vs_weight=0.0)
    model, batch, target, mask = _inputs(cfg, batch_fixture, transform_fixture)
    model.eval()
    members = torch.randn(2, 5, 4, 2).abs()
    model._noise_samples = lambda *a, **k: members
    with torch.no_grad():
        model.sample_head.vs_weight = 0.0
        _, _, l0 = _loss(model, batch, target, mask)
        model.sample_head.vs_weight = 0.5
        out, _, l1 = _loss(model, batch, target, mask)
    scaling = model.sample_head.loss_scaling
    expected = 0.9 * sum(
        scaling[i] * 0.5 * out["loss_vs"][name]
        for i, name in enumerate(model.sample_head.variable_names)
    )
    assert torch.allclose(l1 - l0, expected, atol=1e-5)


def test_vs_on_gradients_and_logs(model_cfg_fixture, batch_fixture, transform_fixture):
    cfg = _noise_on(model_cfg_fixture, vs_weight=0.5)
    model, batch, target, mask = _inputs(cfg, batch_fixture, transform_fixture)
    out, _, loss = _loss(model, batch, target, mask)
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.sample_head, model.oat_block):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert any(g.abs().sum() > 0 for g in grads)
    names = list(cfg.output_head.variable_to_distribution)
    assert set(out["loss_vs"]) == set(names)
    assert all(torch.isfinite(v) for v in out["loss_vs"].values())


@pytest.mark.parametrize("kw", [{"vs_weight": -1.0}, {"vs_p": 0.0}])
def test_vs_config_validation(model_cfg_fixture, kw):
    model_cfg_fixture.query_noise = _qn_cfg(**kw)
    model_cfg_fixture.pre_oat_loss_weight = 0.1
    with pytest.raises(ValueError):
        OAT(model_cfg_fixture)


def test_mixture_nugget_index_is_argmin(model_cfg_fixture):
    cfg = _noise_on(
        model_cfg_fixture, mixing="scale_mixture", length_scales_km=(2.0, 0.1, 8.0)
    )
    head = OAT(cfg).sample_head
    assert head.pi.argmax().item() == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"mixing": "scale_mixture"},  # (2, 8): no scale of 1 km or less
        {"length_scales_km": (0.0, 2.0)},
        {"mixing": "other"},
        {"vs_warmup_epochs": -1},
    ],
)
def test_mixture_config_validation(model_cfg_fixture, kw):
    with pytest.raises(ValueError):
        OAT(_noise_on(model_cfg_fixture, **kw))


def test_default_mixing_keeps_state_dict_keys(model_cfg_fixture):
    keys_default = list(OAT(_noise_on(model_cfg_fixture)).state_dict())
    keys_channels = list(
        OAT(_noise_on(model_cfg_fixture, mixing="channels")).state_dict()
    )
    assert keys_default == keys_channels
    assert not any("mixture" in k for k in keys_default)


def test_warmup_oat_train_no_vs_eval_vs(
    model_cfg_fixture, batch_fixture, transform_fixture
):
    cfg = _noise_on(
        model_cfg_fixture,
        mixing="scale_mixture",
        length_scales_km=(0.1, 8.0),
        vs_weight=0.5,
        vs_warmup_epochs=5,
    )
    model, batch, target, mask = _inputs(cfg, batch_fixture, transform_fixture)
    model.train()
    out, _, loss = _loss(model, batch, target, mask)
    assert torch.isfinite(loss) and "loss_vs" not in out
    model.eval()
    out, _, _ = _loss(model, batch, target, mask)
    assert "loss_vs" in out


def _predictor(cfg, transform_fixture):
    input_transform, output_transform = transform_fixture
    return OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(cfg.output_head.variable_to_distribution),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )


def test_epoch_hook_sets_epoch_and_logs(
    model_cfg_fixture, transform_fixture, monkeypatch
):
    cfg = _noise_on(
        model_cfg_fixture,
        mixing="scale_mixture",
        length_scales_km=(0.1, 8.0),
        vs_weight=0.5,
        vs_warmup_epochs=2,
    )
    pred = _predictor(cfg, transform_fixture)
    monkeypatch.setattr(type(pred), "current_epoch", property(lambda s: 3))
    logged = {}
    monkeypatch.setattr(
        pred, "log", lambda name, value, **kw: logged.__setitem__(name, float(value))
    )
    pred.on_train_epoch_start()
    assert pred.model.sample_head.epoch == 3.0
    assert set(logged) == {"noise_pi_0.1km", "noise_pi_8km", "vs_weight_effective"}
    assert logged["vs_weight_effective"] == 0.5
    assert logged["noise_pi_0.1km"] + logged["noise_pi_8km"] == pytest.approx(1.0)


def test_epoch_hook_without_query_noise(model_cfg_fixture, transform_fixture):
    pred = _predictor(model_cfg_fixture, transform_fixture)
    pred.on_train_epoch_start()  # must not fail
    assert not hasattr(pred.model, "sample_head")
