"""End-to-end field-path integration smoke for all four GP baseline variants."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from oat.data_module.field import Field
from oat.data_module.setup import collate
from oat.predictor_module.gp_predictor import setup_gp_predictor

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


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
                "dataset=cosmo_peakweather/temp",
                *overrides,
            ],
        )

    predictor = setup_gp_predictor(
        cfg,
        list(cfg.dataset.obs_vars),
        (IdentityTransform(), IdentityTransform()),
    )

    batch = collate(
        [
            _make_sample(0, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
            _make_sample(1, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
        ]
    )
    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss)


class ValuesToDoubleTransform:
    """Mimics a real ``NestedTransform`` of ``FieldValueTransform``s after the
    predictor has been cast to double precision: the inner per-field
    transforms are ``nn.Module``s whose buffers get promoted to float64 by
    ``predictor.double()``, so applying them to float32 ``Field.values``
    upcasts the result -- but ``Field.coords`` is passed through untouched
    (see ``FieldValueTransform.forward``), so it stays float32. This is what
    lets ``time_emb`` (derived from ``coords``) end up float32 while every
    other tensor in the batch is float64 under ``trainer.precision=64``.
    """

    def __call__(self, batch: dict) -> dict:
        def cast_group(group: dict) -> dict:
            return {
                "fields": {
                    name: Field(
                        f.coords,
                        f.values.double(),
                        f.var_names,
                        f.mask.double() if f.mask is not None else None,
                    )
                    for name, f in group["fields"].items()
                }
            }

        return {
            k: (cast_group(v) if k in ("observation_data", "query_data") else v)
            for k, v in batch.items()
        }

    def inverse(self, x):
        return x


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
def test_gp_field_training_step_runs_under_precision_64(model_name, overrides):
    """Regression test for a time_emb/coords dtype mismatch under
    trainer.precision=64: Field.coords (and thus the reconstructed time
    embedding) must be cast to match the dtype of the other, already
    transformed tensors in the batch, not left at float32.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "predictor=gp_predictor",
                "dataset=cosmo_peakweather/temp",
                "trainer.precision=64",
                *overrides,
            ],
        )

    predictor = setup_gp_predictor(
        cfg,
        list(cfg.dataset.obs_vars),
        (ValuesToDoubleTransform(), ValuesToDoubleTransform()),
    )
    predictor.double()

    batch = collate(
        [
            _make_sample(0, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
            _make_sample(1, 4, 3, cfg.dataset.nwp_vars, cfg.dataset.topo_vars),
        ]
    )
    loss = predictor.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float64
