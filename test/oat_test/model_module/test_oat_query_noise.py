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
