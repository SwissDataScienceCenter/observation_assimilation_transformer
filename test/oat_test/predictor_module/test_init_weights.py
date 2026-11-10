"""Tests for init_weights_from: start a run from the weights of a checkpoint."""

import copy
from typing import Any

import pytest
import torch
from oat_test.conftest import FIELD_ROLES
from omegaconf import OmegaConf

from oat.model_module.model import QueryNoiseConfig
from oat.predictor_module import OATPredictor
from oat.predictor_module.checkpoint_utils import load_init_weights


def _noise_on(cfg: Any) -> Any:
    cfg = copy.deepcopy(cfg)
    cfg.query_noise = QueryNoiseConfig(
        length_scales_km=(2.0, 8.0),
        km_per_unit=1.0,
        channels_per_scale=2,
        num_frequencies=16,
        mlp_hidden_layers=1,
        train_members=4,
    )
    cfg.pre_oat_loss_weight = 0.1
    return cfg


def _predictor(cfg: Any, transforms: Any, seed: int) -> OATPredictor:
    torch.manual_seed(seed)
    input_transform, output_transform = transforms
    return OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(cfg.output_head.variable_to_distribution.keys()),
    )


def _cfg(init: Any = None, resume: Any = None) -> Any:
    return OmegaConf.create(
        {"init_weights_from": init, "resume_from_checkpoint": resume}
    )


def _save(pred: OATPredictor, path: Any) -> str:
    torch.save({"state_dict": pred.state_dict()}, path)
    return str(path)


def test_init_weights_none_is_noop(model_cfg_fixture, transform_fixture):
    pred = _predictor(_noise_on(model_cfg_fixture), transform_fixture, 0)
    before = {k: v.clone() for k, v in pred.state_dict().items()}
    load_init_weights(pred, _cfg())
    for key, value in pred.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_init_weights_loads_all_keys(model_cfg_fixture, transform_fixture, tmp_path):
    cfg = _noise_on(model_cfg_fixture)
    source = _predictor(cfg, transform_fixture, 0)
    target = _predictor(cfg, transform_fixture, 1)
    path = _save(source, tmp_path / "stage1.ckpt")
    load_init_weights(target, _cfg(init=path))
    expected = source.state_dict()
    for key, value in target.state_dict().items():
        assert torch.equal(value, expected[key]), key


def test_init_weights_strict_mismatch_raises(
    model_cfg_fixture, transform_fixture, tmp_path
):
    source = _predictor(model_cfg_fixture, transform_fixture, 0)  # no query noise
    target = _predictor(_noise_on(model_cfg_fixture), transform_fixture, 1)
    path = _save(source, tmp_path / "stage1.ckpt")
    with pytest.raises(RuntimeError, match="sample_head"):
        load_init_weights(target, _cfg(init=path))


def test_init_weights_with_resume_raises(model_cfg_fixture, transform_fixture):
    pred = _predictor(_noise_on(model_cfg_fixture), transform_fixture, 0)
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        load_init_weights(pred, _cfg(init="a.ckpt", resume="b.ckpt"))
