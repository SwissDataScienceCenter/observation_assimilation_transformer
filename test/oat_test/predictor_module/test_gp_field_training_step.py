"""End-to-end field-path integration smoke for all four GP baseline variants."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from oat.data_module.field import Field
from oat.data_module.field_setup import field_collate
from oat.predictor_module.gp_predictor import setup_gp_predictor

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")
FIELD_EXTENT = {"min": [2255500.0, 840500.0], "max": [2964500.0, 1479500.0]}


class IdentityTransform:
    def __call__(self, batch):
        return batch

    def inverse(self, x):
        return x


def _make_sample(idx: int, n_ctx: int, n_qry: int, nwp_vars, topo_vars) -> dict:
    def grp(n: int) -> dict:
        coords = torch.zeros(n, 3)
        coords[:, 0] = torch.linspace(0, 1, n)
        coords[:, 1] = torch.linspace(0, 1, n)
        coords[:, 2] = float(idx) + 100.0
        return {
            "obs": Field(
                coords.clone(), torch.randn(n, 1), ["temperature"], torch.ones(n, 1)
            ),
            "nwp": Field(coords.clone(), torch.randn(n, len(nwp_vars)), list(nwp_vars)),
            "geo": Field(coords.clone(), torch.randn(n, 3), ["x", "y", "d"]),
            "topo": Field(
                coords.clone(), torch.randn(n, len(topo_vars)), list(topo_vars)
            ),
        }

    return {
        "index": idx,
        "observation_data": {"fields": grp(n_ctx)},
        "query_data": {"fields": grp(n_qry)},
    }


@pytest.mark.parametrize(
    "model_name,overrides",
    [
        ("gp_baseline", ["model/topographic_encoder=mlp"]),
        ("deep_gp_baseline", ["model/topographic_encoder=mlp"]),
        (
            "gp_oat_baseline",
            ["model/global_encoder=point", "model/topographic_encoder=mlp"],
        ),
        (
            "deep_gp_oat_baseline",
            ["model/global_encoder=point", "model/topographic_encoder=mlp"],
        ),
    ],
)
def test_gp_field_training_step_runs(model_name, overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "predictor=gp_predictor",
                "dataset=cosmo_peakweather/field_temp",
                *overrides,
            ],
        )

    predictor = setup_gp_predictor(
        cfg,
        FIELD_EXTENT,
        list(cfg.dataset.obs_vars),
        (IdentityTransform(), IdentityTransform()),
    )

    batch = field_collate(
        [
            _make_sample(0, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
            _make_sample(1, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
        ]
    )
    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss)
