"""End-to-end field-path integration smoke: tiny OAT trained via one training_step.

This is the capstone test for the field-model interface (Tasks 1-9): it builds a
minimal but complete OAT/UninformedOAT model from field-shaped configs, collates
synthetic Field-based samples, and runs a real ``OATPredictor.training_step`` /
``_normalize_batch`` through the model, asserting a finite loss.
"""

import os

import pytest
import torch

from oat.data_module.field import Field
from oat.data_module.setup import collate
from oat.data_module.transform import (
    FieldValueTransform,
    IdentityTransform,
    NestedTransform,
)
from oat.model_module import model as M
from oat.predictor_module.oat_predictor import OATPredictor

D_MODEL = 16
FIELD_ROLES = {"target": "obs", "global": "nwp", "geo_bias": "geo", "topo": "topo"}


def _make_sample(idx: int, n_ctx: int, n_qry: int) -> dict:
    def grp(n: int) -> dict:
        coords = torch.zeros(n, 3)
        coords[:, 0] = torch.linspace(0, 1, n)
        coords[:, 1] = torch.linspace(0, 1, n)
        coords[:, 2] = float(idx) + 100.0
        return {
            "obs": Field(
                coords.clone(), torch.randn(n, 1), ["temperature"], torch.ones(n, 1)
            ),
            "nwp": Field(coords.clone(), torch.randn(n, 2), ["T_2M", "T_G"]),
            "geo": Field(coords.clone(), torch.randn(n, 3), ["x", "y", "d"]),
            "topo": Field(coords.clone(), torch.randn(n, 5), list("pqrst")),
        }

    return {
        "index": idx,
        "observation_data": {"fields": grp(n_ctx)},
        "query_data": {"fields": grp(n_qry)},
    }


def _leaf_transform() -> NestedTransform:
    def leaves() -> dict:
        return {
            n: FieldValueTransform(IdentityTransform())
            for n in ("obs", "nwp", "geo", "topo")
        }

    return NestedTransform(
        {
            "observation_data": {"fields": leaves()},
            "query_data": {"fields": leaves()},
        }
    )


def _backbone_cfgs() -> (
    tuple[M.GlobalEncoderConfig, M.TopographicEncoderConfig, M.NwpTopoFusionConfig]
):
    global_encoder_cfg = M.GlobalEncoderConfig(
        in_channels=2,
        type="Point",
        out_channels=D_MODEL // 2,
        hidden_dims=(),
        act_fn="GELU",
    )
    topographic_encoder_cfg = M.TopographicEncoderConfig(
        in_channels=5,
        out_channels=D_MODEL // 2,
        type="MLP",
        hidden_dims=(),
        act_fn="GELU",
    )
    nwp_topo_fusion_cfg = M.NwpTopoFusionConfig(
        cond_dim=4, hidden_channels=D_MODEL, out_channels=D_MODEL
    )
    return global_encoder_cfg, topographic_encoder_cfg, nwp_topo_fusion_cfg


def _oat_config() -> M.OATConfig:
    global_encoder_cfg, topographic_encoder_cfg, nwp_topo_fusion_cfg = _backbone_cfgs()

    masking_cfg = M.MaskingConfig(mode="knn", k=2)
    transformer_cfg = M.TransformerConfig(
        d_geo_bias=3, d_model=D_MODEL, dim_feedforward=2 * D_MODEL
    )
    innovation_fusion_cfg = M.InnovationFusionConfig(
        cond_dim=4, hidden_channels=D_MODEL, out_channels=D_MODEL
    )
    output_head_cfg = M.OutputHeadConfig(
        in_features=D_MODEL, variable_to_distribution={"temperature": "normal"}
    )

    return M.OATConfig(
        global_encoder=global_encoder_cfg,
        topographic_encoder=topographic_encoder_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        innovation_fusion=innovation_fusion_cfg,
        masking=masking_cfg,
        transformer=transformer_cfg,
        output_head=output_head_cfg,
        num_samples=4,
    )


def _uninformed_oat_config() -> M.UninformedOATConfig:
    global_encoder_cfg, topographic_encoder_cfg, nwp_topo_fusion_cfg = _backbone_cfgs()
    output_head_cfg = M.OutputHeadConfig(
        in_features=D_MODEL, variable_to_distribution={"temperature": "normal"}
    )
    return M.UninformedOATConfig(
        global_encoder=global_encoder_cfg,
        topographic_encoder=topographic_encoder_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        output_head=output_head_cfg,
        num_samples=4,
    )


def _build_predictor(
    oat_model_cfg: M.OATConfig | M.UninformedOATConfig,
) -> OATPredictor:
    return OATPredictor(
        oat_model_cfg=oat_model_cfg,
        input_transform=_leaf_transform(),
        output_transform=IdentityTransform(),
        output_var_names=["temperature"],
        field_roles=FIELD_ROLES,
    )


def test_field_training_step_runs():
    predictor = _build_predictor(_oat_config())
    batch = collate([_make_sample(0, 4, 3), _make_sample(1, 4, 3)])
    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_field_training_step_uninformed_runs():
    predictor = _build_predictor(_uninformed_oat_config())
    batch = collate([_make_sample(0, 4, 3), _make_sample(1, 4, 3)])
    batch = predictor._normalize_batch(batch)
    out = predictor.model.forward(
        global_data=batch["global_data"], query_data=batch["query_data"]
    )
    assert torch.isfinite(out["samples"]).all()


def test_temp_config_composes():
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cosmo_peakweather/temp",
                "model/global_encoder=point",
                "model/topographic_encoder=mlp",
            ],
        )
    assert cfg.dataset.roles["global"] == "nwp"
    assert list(cfg.dataset.obs_vars) == ["temperature"]


_DATA_DIR = "/scratch/mch/cdonner/data/obs_downscaling/raw/cosmo-1e_rechunked"
_DATA = os.path.join(_DATA_DIR, "nwp_extracted.zarr")


@pytest.mark.skipif(not os.path.exists(_DATA), reason="COSMO data not present")
def test_field_training_step_real_data_smoke():
    import logging

    from hydra import compose, initialize_config_dir

    from oat.data_module.setup import setup_data_and_transform
    from oat.predictor_module.oat_predictor import setup_oat_cfg, setup_oat_predictor

    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cosmo_peakweather/temp",
                "model/global_encoder=point",
                "model/topographic_encoder=mlp",
                "dataloader.train.batch_size=2",
                "dataloader.train.num_workers=0",
                "dataloader.train.prefetch_factor=null",
                "dataloader.train.persistent_workers=false",
            ],
        )

    logger = logging.getLogger("test_field_training_step_real_data_smoke")
    (
        _dataset,
        _n_train,
        train_loader,
        _val_loader,
        _test_loader,
        input_transform,
        output_transform,
    ) = setup_data_and_transform(cfg, logger)

    batch = next(iter(train_loader))

    oat_model_cfg = setup_oat_cfg(cfg, logger)
    predictor = setup_oat_predictor(
        oat_model_cfg,
        estimated_total_step=1,
        var_names=cfg.dataset.obs_vars,
        transforms=(input_transform, output_transform),
        cfg=cfg,
    )

    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss)
