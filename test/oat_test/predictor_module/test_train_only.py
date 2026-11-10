"""Tests for train_only: train only the query-noise sample head (stage 2)."""

from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from oat_test.conftest import FIELD_ROLES
from torch.utils.data import DataLoader

from oat.model_module.model import QueryNoiseConfig
from oat.predictor_module import OATPredictor

HEAD = "model.sample_head."


def _noise_cfg(cfg: Any) -> Any:
    cfg.query_noise = QueryNoiseConfig(
        length_scales_km=(2.0, 8.0),
        km_per_unit=1.0,
        channels_per_scale=2,
        num_frequencies=16,
        mlp_hidden_layers=1,
        train_members=4,
    )
    cfg.pre_oat_loss_weight = 0.1
    # A hidden layer with BatchNorm, as in the real checkpoints.
    cfg.global_encoder.hidden_dims = (8,)
    cfg.global_encoder.use_batchnorm = True
    return cfg


def _predictor(cfg: Any, transforms: Any, train_only: str | None) -> OATPredictor:
    input_transform, output_transform = transforms
    return OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(cfg.output_head.variable_to_distribution.keys()),
        lr=1e-2,
        optimizer_class=torch.optim.AdamW,
        train_only=train_only,
    )


def _frozen_modules(pred: OATPredictor) -> list[torch.nn.Module]:
    head = set(pred.model.sample_head.modules())
    return [
        m
        for m in pred.modules()
        if m is not pred and m is not pred.model and m not in head
    ]


def _optimizer(pred: OATPredictor) -> torch.optim.Optimizer:
    out = pred.configure_optimizers()
    return out["optimizer"] if isinstance(out, dict) else out


@pytest.fixture
def stage2(model_cfg_fixture, transform_fixture) -> OATPredictor:
    torch.manual_seed(0)
    return _predictor(_noise_cfg(model_cfg_fixture), transform_fixture, "sample_head")


def test_train_only_freezes_all_but_sample_head(stage2):
    for name, p in stage2.named_parameters():
        assert p.requires_grad == name.startswith(HEAD), name


def test_train_only_none_changes_nothing(model_cfg_fixture, transform_fixture):
    pred = _predictor(_noise_cfg(model_cfg_fixture), transform_fixture, None)
    assert all(p.requires_grad for p in pred.parameters())


def test_train_only_unknown_value_raises(model_cfg_fixture, transform_fixture):
    with pytest.raises(ValueError, match="train_only"):
        _predictor(_noise_cfg(model_cfg_fixture), transform_fixture, "decoder")


def test_train_only_without_query_noise_raises(model_cfg_fixture, transform_fixture):
    with pytest.raises(ValueError, match="query noise"):
        _predictor(model_cfg_fixture, transform_fixture, "sample_head")


def test_train_call_keeps_frozen_modules_in_eval(stage2):
    stage2.train()
    assert not any(m.training for m in _frozen_modules(stage2))
    assert all(m.training for m in stage2.model.sample_head.modules())
    stage2.eval()
    assert not any(m.training for m in stage2.modules())


def test_optimizer_holds_only_head_parameters(stage2):
    ids = {id(p) for g in _optimizer(stage2).param_groups for p in g["params"]}
    assert ids == {id(p) for p in stage2.model.sample_head.parameters()}


def test_one_step_changes_only_head(stage2, batch_fixture):
    before = {k: v.clone() for k, v in stage2.state_dict().items()}
    optimizer = _optimizer(stage2)
    stage2.train()
    loss = stage2.training_step(batch_fixture, 0)
    loss.backward()
    optimizer.step()
    after = stage2.state_dict()
    bn_keys = [k for k in before if "running_mean" in k or "running_var" in k]
    assert bn_keys, "the test model must have a BatchNorm"
    for key, value in before.items():
        if not key.startswith(HEAD):
            assert torch.equal(value, after[key]), key
    assert any(
        not torch.equal(before[k], after[k]) for k in before if k.startswith(HEAD)
    )


class _ModeRecorder(pl.Callback):
    def __init__(self) -> None:
        self.batches = 0
        self.frozen_in_train = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self.batches += 1
        if any(m.training for m in _frozen_modules(pl_module)):
            self.frozen_in_train = True


def _fit(pred: OATPredictor, batch: Any, tmp_path: Any) -> pl.Trainer:
    # The list dataset is not a station dataset. Skip the support check.
    pred.validate_output_head_supports = lambda dataset: None  # type: ignore[method-assign]
    loader = DataLoader([batch], batch_size=None)
    recorder = _ModeRecorder()
    trainer = pl.Trainer(
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        default_root_dir=tmp_path,
        callbacks=[recorder],
    )
    trainer.fit(pred, train_dataloaders=loader, val_dataloaders=loader)
    trainer.recorder = recorder  # type: ignore[attr-defined]
    return trainer


def test_eval_mode_survives_trainer_epochs(stage2, batch_fixture, tmp_path):
    recorder = _fit(stage2, batch_fixture, tmp_path).recorder
    assert recorder.batches >= 2
    assert not recorder.frozen_in_train


def test_checkpoint_round_trip_with_train_only(stage2, batch_fixture, tmp_path):
    trainer = _fit(stage2, batch_fixture, tmp_path)
    path = tmp_path / "stage2.ckpt"
    trainer.save_checkpoint(path)
    reloaded = OATPredictor.load_from_checkpoint(path).eval()
    assert reloaded.train_only == "sample_head"
    samples = reloaded.predict_step(batch_fixture, 0, n_samples=3)
    assert torch.isfinite(samples).all()
