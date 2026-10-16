"""Shared field-shaped fixtures for the OAT predictor and model tests.

The fixtures build a small synthetic batch in code. They use the same field
layout as the live datasets: an ``obs`` target field, an ``nwp`` global field,
a ``geo`` geo-bias field and a ``topo`` field.
"""

from typing import Any

import pytest
import torch

from oat.data_module.field import Field
from oat.data_module.setup import collate
from oat.data_module.transform import (
    FieldValueTransform,
    IdentityTransform,
    NestedTransform,
)
from oat.model_module.model import (
    GlobalEncoderConfig,
    InnovationFusionConfig,
    LearnableGatingConfig,
    MaskingConfig,
    NwpTopoFusionConfig,
    OATConfig,
    OutputHeadConfig,
    TopographicEncoderConfig,
    TransformerConfig,
    UninformedOATConfig,
)

D_MODEL = 8
N_NWP = 2
N_GEO = 3
N_TOPO = 5
OBS_VARS = ["var_0", "var_1"]
FIELD_ROLES = {"target": "obs", "global": "nwp", "geo_bias": "geo", "topo": "topo"}


def _group(idx: int, n: int) -> dict[str, Field]:
    coords = torch.zeros(n, 3)
    coords[:, 0] = torch.linspace(0, 1, n)
    coords[:, 1] = torch.linspace(0, 1, n)
    coords[:, 2] = float(idx) + 100.0
    n_obs = len(OBS_VARS)
    return {
        "obs": Field(
            coords.clone(), torch.randn(n, n_obs), list(OBS_VARS), torch.ones(n, n_obs)
        ),
        "nwp": Field(coords.clone(), torch.randn(n, N_NWP), ["T_2M", "T_G"]),
        "geo": Field(coords.clone(), torch.randn(n, N_GEO), ["x", "y", "d"]),
        "topo": Field(coords.clone(), torch.randn(n, N_TOPO), list("pqrst")),
    }


def _make_sample(idx: int, n_ctx: int, n_qry: int) -> dict[str, Any]:
    return {
        "index": idx,
        "observation_data": {"fields": _group(idx, n_ctx)},
        "query_data": {"fields": _group(idx, n_qry)},
    }


def _backbone_cfgs() -> (
    tuple[GlobalEncoderConfig, TopographicEncoderConfig, NwpTopoFusionConfig]
):
    global_enc_cfg = GlobalEncoderConfig(
        in_channels=N_NWP,
        type="Point",
        out_channels=D_MODEL // 2,
        hidden_dims=(),
    )
    topography_enc_cfg = TopographicEncoderConfig(
        in_channels=N_TOPO,
        out_channels=D_MODEL // 2,
        type="MLP",
        hidden_dims=(),
    )
    nwp_topo_fusion_cfg = NwpTopoFusionConfig(
        cond_dim=4, out_channels=D_MODEL, hidden_channels=D_MODEL
    )
    return global_enc_cfg, topography_enc_cfg, nwp_topo_fusion_cfg


def _output_head_cfg() -> OutputHeadConfig:
    return OutputHeadConfig(
        in_features=D_MODEL,
        variable_to_distribution={v: "normal" for v in OBS_VARS},
        loss_scaling=None,
        crps_mc_samples=50,
        observation_assimilation_as="innovation",
    )


@pytest.fixture
def batch_fixture() -> dict[str, Any]:
    """Collated field batch: 2 samples, 6 context points, 5 query points."""
    torch.manual_seed(0)
    return collate([_make_sample(0, 6, 5), _make_sample(1, 6, 5)])


@pytest.fixture
def model_cfg_fixture() -> OATConfig:
    global_enc_cfg, topography_enc_cfg, nwp_topo_fusion_cfg = _backbone_cfgs()
    return OATConfig(
        global_encoder=global_enc_cfg,
        topographic_encoder=topography_enc_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        innovation_fusion=InnovationFusionConfig(
            cond_dim=4, out_channels=D_MODEL, hidden_channels=D_MODEL
        ),
        transformer=TransformerConfig(
            d_geo_bias=N_GEO,
            d_model=D_MODEL,
            dim_feedforward=2 * D_MODEL,
            dim_geo_bias_feedforward=4,
        ),
        masking=MaskingConfig(mode="knn", k=2),
        output_head=_output_head_cfg(),
        learnable_gating=LearnableGatingConfig(
            in_channels=len(OBS_VARS), out_channels=D_MODEL, hidden_channels=4
        ),
        num_samples=10,
    )


@pytest.fixture
def uninformed_model_cfg_fixture() -> UninformedOATConfig:
    global_enc_cfg, topography_enc_cfg, nwp_topo_fusion_cfg = _backbone_cfgs()
    return UninformedOATConfig(
        global_encoder=global_enc_cfg,
        topographic_encoder=topography_enc_cfg,
        nwp_topo_fusion=nwp_topo_fusion_cfg,
        output_head=_output_head_cfg(),
        num_samples=10,
    )


@pytest.fixture
def transform_fixture() -> tuple[NestedTransform, IdentityTransform]:
    """Identity input transform on every field, identity output transform."""

    def leaves() -> dict[str, FieldValueTransform]:
        return {
            n: FieldValueTransform(IdentityTransform())
            for n in ("obs", "nwp", "geo", "topo")
        }

    input_transform = NestedTransform(
        {
            "observation_data": {"fields": leaves()},
            "query_data": {"fields": leaves()},
        }
    )
    return input_transform, IdentityTransform()
