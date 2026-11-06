import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from oat_test.conftest import FIELD_ROLES, OBS_VARS
from torch.distributions import constraints

from oat.data_module.support import VariableSupport
from oat.data_module.transform import IdentityTransform
from oat.model_module.distributions import _ALL_DISTRIBUTIONS
from oat.model_module.fusion_layer import ObservationEmbedding
from oat.model_module.model import (
    ObservationEmbeddingConfig,
    OATConfig,
    QueryNoiseConfig,
)
from oat.predictor_module import OATPredictor
from scripts.train_model import setup_oat_cfg


@pytest.mark.parametrize("compute_loss", [True, False])
def test_simple_predictor_forward(
    batch_fixture, transform_fixture, compute_loss, model_cfg_fixture
):

    input_transform, output_transform = transform_fixture

    if compute_loss:
        model_cfg_fixture.pre_oat_loss_weight = 0.2

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    # forward expects a normalized and unwrapped batch.
    batch = pred._normalize_batch(batch_fixture)
    if not compute_loss:
        out = pred(batch, return_loss_and_targets=False)
    else:
        out, per_var_losses, loss, target, target_mask = pred.forward(
            batch, return_loss_and_targets=True
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

        sum = 0.0
        for per_var_loss in per_var_losses.values():
            assert isinstance(per_var_loss, torch.Tensor)
            assert per_var_loss.shape == ()

            sum += per_var_loss

        assert torch.isclose(sum, loss)


@pytest.mark.parametrize("compile", [True, False], ids=["compile", "no_compile"])
@pytest.mark.parametrize(
    "observation_assimilation_as",
    ["innovation", "target"],
    ids=["innovation", "target"],
)
def test_simple_predictor_train_step(
    batch_fixture,
    transform_fixture,
    model_cfg_fixture,
    compile,
    observation_assimilation_as,
):

    if observation_assimilation_as == "innovation":
        model_cfg_fixture.output_head.observation_assimilation_as = "innovation"
    else:
        model_cfg_fixture.output_head.observation_assimilation_as = "target"

    output_var_names = list(
        model_cfg_fixture.output_head.variable_to_distribution.keys()
    )
    input_transform, output_transform = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=output_var_names,
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    if compile:
        pred = torch.compile(pred)

    pred.training_step(batch=batch_fixture, batch_idx=None)


def test_uninformed_predictor_train_step(
    batch_fixture, transform_fixture, uninformed_model_cfg_fixture
):

    input_transform, output_transform = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=uninformed_model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(
            uninformed_model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    pred.training_step(batch=batch_fixture, batch_idx=None)


def test_checkpoint(batch_fixture, transform_fixture, model_cfg_fixture):

    output_var_names = list(
        model_cfg_fixture.output_head.variable_to_distribution.keys()
    )
    input_transform, output_transform = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=output_var_names,
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = f"{tmpdir}/model.ckpt"

        # Manually create a Lightning checkpoint
        torch.save(
            {
                "state_dict": pred.state_dict(),
                "hyper_parameters": pred.hparams,
                "pytorch-lightning_version": pl.__version__,
            },
            ckpt_path,
        )

        # Reload the module
        reloaded_pred = type(pred).load_from_checkpoint(ckpt_path)

    # Remove randomness like dropouts
    pred = pred.eval()
    reloaded_pred = reloaded_pred.eval()

    # Verify they are different instances
    assert pred is not reloaded_pred

    pl.seed_everything(42)
    prediction = pred(pred._normalize_batch(batch_fixture))

    pl.seed_everything(42)
    prediction_reloaded = reloaded_pred(reloaded_pred._normalize_batch(batch_fixture))
    # Check that all outputs apart from the samples are equivalent before and
    # after checkpointing

    def same_distribution(d1, d2):
        if type(d1) != type(d2):
            return False

        params1 = d1.__dict__
        params2 = d2.__dict__

        if params1.keys() != params2.keys():
            return False

        for k in params1:
            v1, v2 = params1[k], params2[k]
            if torch.is_tensor(v1):
                if not torch.allclose(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    # assert predicted distributions are the same (same parametrization)
    for var_name in output_var_names:
        assert same_distribution(prediction[var_name], prediction_reloaded[var_name])

        assert same_distribution(
            prediction["pre_oat_output"][var_name],
            prediction_reloaded["pre_oat_output"][var_name],
        )
    # assert that the samples, given that we forced a seed, are the same before and after checkpointing
    assert torch.allclose(
        prediction_reloaded["pre_oat_output"]["samples"],
        prediction["pre_oat_output"]["samples"],
    )


def test_default_hydra_config(transform_fixture):
    # this test checks if the default hydra config can be used to instantiate a predictor (without using the cfg fixture)
    input_transform, output_transform = transform_fixture
    config_dir = str(Path(__file__).resolve().parents[3] / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config"
        )  # TODO potentially use all configs in this dir, not just config

    logger = logging.getLogger(__name__)
    oat_cfg = setup_oat_cfg(cfg, logger)

    assert isinstance(oat_cfg, OATConfig)
    assert oat_cfg.output_head.in_features == cfg.model.d_model
    assert oat_cfg.observation_embedding is not None
    assert oat_cfg.output_head.variable_to_distribution == dict(
        cfg.dataset.obs_variable_to_distribution
    )
    assert set(oat_cfg.output_head.variable_to_distribution.keys()) == set(
        cfg.dataset.obs_vars
    )

    pred = OATPredictor(
        oat_model_cfg=oat_cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=cfg.dataset.obs_vars,
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )


def test_radius_masking_rejected_with_sphere_projection():
    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["model/masking=radius"],
        )
    logger = logging.getLogger(__name__)

    def _sphere_projection(coords: torch.Tensor) -> torch.Tensor:
        lon = torch.deg2rad(coords[..., 0])
        lat = torch.deg2rad(coords[..., 1])
        cos_lat = torch.cos(lat)
        return torch.stack(
            [cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)],
            dim=-1,
        )

    with pytest.raises(ValueError, match="not compatible with an index-coordinate"):
        setup_oat_cfg(
            cfg,
            logger,
            project_index_coords=_sphere_projection,
        )


def _make_train_dataset_stub(obs_vars: list[str], obs_support: VariableSupport):
    # Mirrors TrainDataset.dataset.obs_var_support and .config.obs_vars access pattern.
    return SimpleNamespace(
        dataset=SimpleNamespace(
            obs_var_support=obs_support,
            config=SimpleNamespace(obs_vars=obs_vars),
        )
    )


def test_validate_output_head_support_passes_for_compatible_supports(
    transform_fixture,
    model_cfg_fixture,
):
    input_transform, _ = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=IdentityTransform(),
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    obs_vars = list(model_cfg_fixture.output_head.variable_to_distribution.keys())
    n_vars = len(obs_vars)
    train_dataset = _make_train_dataset_stub(
        obs_vars=obs_vars,
        obs_support=VariableSupport(
            lower_bounds=torch.full((n_vars,), -1.0),
            upper_bounds=torch.full((n_vars,), 1.0),
            lower_open=torch.zeros(n_vars, dtype=torch.bool),
            upper_open=torch.zeros(n_vars, dtype=torch.bool),
        ),
    )

    # Should not raise.
    pred.validate_output_head_supports(train_dataset)


def test_validate_output_head_support_raises_with_meaningful_message(
    transform_fixture,
    model_cfg_fixture,
):
    input_transform, _ = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=IdentityTransform(),
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    obs_vars = list(model_cfg_fixture.output_head.variable_to_distribution.keys())
    first_var = obs_vars[0]
    # Make first variable stricter than transformed support.
    pred.model.output_head.distribution_mappers[first_var].support = (
        constraints.nonnegative
    )

    n_vars = len(obs_vars)
    lower = torch.zeros(n_vars)
    lower[0] = -2.0
    upper = torch.ones(n_vars)

    train_dataset = _make_train_dataset_stub(
        obs_vars=obs_vars,
        obs_support=VariableSupport(
            lower_bounds=lower,
            upper_bounds=upper,
            lower_open=torch.zeros(n_vars, dtype=torch.bool),
            upper_open=torch.zeros(n_vars, dtype=torch.bool),
        ),
    )

    with pytest.raises(ValueError, match=f"{first_var}.*nonnegative.*-2"):
        pred.validate_output_head_supports(train_dataset)


def test_validate_output_head_support_raises_type_error_for_invalid_dataset(
    transform_fixture,
    model_cfg_fixture,
):
    input_transform, _ = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=IdentityTransform(),
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    with pytest.raises(TypeError, match="TrainDataset-like.*obs_var_support.*obs_vars"):
        pred.validate_output_head_supports(object())


def _distribution_class_by_name() -> dict[str, type]:
    return {
        dist_cls.name: dist_cls
        for dist_cls in _ALL_DISTRIBUTIONS
        if dist_cls.name is not None
    }


@pytest.mark.parametrize(
    "dist_name,first_var_lower,should_pass",
    [
        ("normal", -2.0, True),
        ("lognormal", 0.1, True),
        ("exponential", -0.1, False),
        ("zero_inflated_lognormal", -0.1, False),
    ],
)
def test_validate_output_head_support_selected_all_distributions_cases(
    transform_fixture,
    model_cfg_fixture,
    dist_name,
    first_var_lower,
    should_pass,
):
    input_transform, _ = transform_fixture

    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=IdentityTransform(),
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )

    dist_by_name = _distribution_class_by_name()
    assert (
        dist_name in dist_by_name
    ), f"Expected '{dist_name}' to be registered in _ALL_DISTRIBUTIONS"

    obs_vars = list(model_cfg_fixture.output_head.variable_to_distribution.keys())
    first_var = obs_vars[0]

    # Override support to exercise predictor validation against selected distributions.
    pred.model.output_head.distribution_mappers[first_var].support = dist_by_name[
        dist_name
    ].support
    pred.model.output_head.variable_to_distribution[first_var] = dist_name

    n_vars = len(obs_vars)
    lower = torch.full((n_vars,), 0.1)
    lower[0] = first_var_lower
    upper = torch.full((n_vars,), 2.0)

    train_dataset = _make_train_dataset_stub(
        obs_vars=obs_vars,
        obs_support=VariableSupport(
            lower_bounds=lower,
            upper_bounds=upper,
            lower_open=torch.zeros(n_vars, dtype=torch.bool),
            upper_open=torch.zeros(n_vars, dtype=torch.bool),
        ),
    )

    if should_pass:
        pred.validate_output_head_supports(train_dataset)
    else:
        with pytest.raises(ValueError, match=f"{first_var}.*{dist_name}"):
            pred.validate_output_head_supports(train_dataset)


def test_predict_step_samples_shape(
    batch_fixture, transform_fixture, model_cfg_fixture
):
    input_transform, output_transform = transform_fixture
    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
    )
    n_samples = 5
    B, N_q = batch_fixture["query_data"]["fields"]["obs"].coords.shape[:2]
    n_vars = len(pred.output_var_names)

    samples = pred.predict_step(batch_fixture, batch_idx=0, n_samples=n_samples)

    assert samples.shape == (B, N_q, n_samples, n_vars)


def test_predict_step_pre_oat_samples_shape(
    batch_fixture, transform_fixture, model_cfg_fixture
):
    input_transform, output_transform = transform_fixture
    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(
            model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
    )
    n_samples = 5
    B, N_q = batch_fixture["query_data"]["fields"]["obs"].coords.shape[:2]
    n_vars = len(pred.output_var_names)

    samples = pred.predict_step(
        batch_fixture, batch_idx=0, n_samples=n_samples, stage="pre_oat"
    )
    assert samples.shape == (B, N_q, n_samples, n_vars)


def test_predict_step_pre_oat_raises_for_uninformed(
    batch_fixture, transform_fixture, uninformed_model_cfg_fixture
):
    input_transform, output_transform = transform_fixture
    pred = OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=uninformed_model_cfg_fixture,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(
            uninformed_model_cfg_fixture.output_head.variable_to_distribution.keys()
        ),
    )
    with pytest.raises(ValueError, match="pre_oat"):
        pred.predict_step(batch_fixture, batch_idx=0, n_samples=3, stage="pre_oat")


def _make_oat_predictor(model_cfg, transform_fixture) -> OATPredictor:
    input_transform, output_transform = transform_fixture
    return OATPredictor(
        field_roles=FIELD_ROLES,
        oat_model_cfg=model_cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=list(model_cfg.output_head.variable_to_distribution.keys()),
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )


@pytest.mark.parametrize("assim_as", ["innovation", "target"])
def test_predictor_train_step_with_observation_embedding(
    batch_fixture, transform_fixture, model_cfg_fixture, assim_as
):
    model_cfg_fixture.output_head.observation_assimilation_as = assim_as
    model_cfg_fixture.observation_embedding = ObservationEmbeddingConfig(
        n_vars=len(OBS_VARS), d_embed=4, hidden_channels=8
    )
    pred = _make_oat_predictor(model_cfg_fixture, transform_fixture)
    assert isinstance(pred.model.obs_embedding, ObservationEmbedding)

    batch = pred._normalize_batch(batch_fixture)
    # Partial observation: var_1 missing at half of the context stations.
    n_ctx = batch["observation_data"]["mask_obs"].shape[1]
    batch["observation_data"]["mask_obs"][:, : n_ctx // 2, 1] = 0.0
    _, _, loss, _, _ = pred.forward(batch, return_loss_and_targets=True)
    loss.backward()
    assert torch.isfinite(loss)
    grad = pred.model.obs_embedding.missing_token.grad
    assert grad is not None and grad[1].abs().sum() > 0  # token of var_1 is used
    assert grad[0].abs().sum() == 0  # var_0 is never missing


@pytest.mark.parametrize("use_embedding", [False, True])
def test_missing_obs_differs_from_observed_zero_only_with_embedding(
    batch_fixture, transform_fixture, model_cfg_fixture, use_embedding
):
    model_cfg_fixture.output_head.observation_assimilation_as = "target"
    model_cfg_fixture.learnable_gating = None
    if use_embedding:
        model_cfg_fixture.observation_embedding = ObservationEmbeddingConfig(
            n_vars=len(OBS_VARS), d_embed=4, hidden_channels=8
        )
    pred = _make_oat_predictor(model_cfg_fixture, transform_fixture).eval()
    batch = pred._normalize_batch(batch_fixture)
    obs_data = batch["observation_data"]
    obs_data["obs"][..., 1] = 0.0

    outputs = []
    for mask_value in (1.0, 0.0):
        obs_data["mask_obs"][..., 1] = mask_value
        with torch.no_grad():
            out, _ = pred.model._forward(
                global_data=batch["global_data"],
                observation_data=obs_data,
                query_data=batch["query_data"],
            )
        outputs.append(out)

    if use_embedding:
        assert not torch.allclose(outputs[0], outputs[1])
    else:
        # Current zero-fill behaviour: "missing" equals "observed 0".
        assert torch.equal(outputs[0], outputs[1])


def _compose_cfg(overrides: list[str]):
    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def test_hydra_config_with_observation_embedding(transform_fixture):
    input_transform, output_transform = transform_fixture
    cfg = _compose_cfg(["model/observation_embedding=default"])
    oat_cfg = setup_oat_cfg(cfg, logging.getLogger(__name__))
    assert oat_cfg.observation_embedding == ObservationEmbeddingConfig(
        n_vars=len(cfg.dataset.obs_vars),
        d_embed=8,
        hidden_channels=16,
        activation="GELU",
    )
    pred = OATPredictor(
        oat_model_cfg=oat_cfg,
        input_transform=input_transform,
        output_transform=output_transform,
        output_var_names=cfg.dataset.obs_vars,
        lr=1e-3,
        optimizer_class=torch.optim.AdamW,
    )
    assert isinstance(pred.model.obs_embedding, ObservationEmbedding)


def test_setup_oat_cfg_without_observation_embedding_key():
    # Copy an old checkpoint config, which has no observation_embedding key.
    cfg = _compose_cfg([])
    with open_dict(cfg.model):
        cfg.model.pop("observation_embedding", None)
    oat_cfg = setup_oat_cfg(cfg, logging.getLogger(__name__))
    assert oat_cfg.observation_embedding is None


def test_nan_at_masked_obs_gives_finite_gradients_in_innovation_mode(
    batch_fixture, transform_fixture, model_cfg_fixture
):
    model_cfg_fixture.output_head.observation_assimilation_as = "innovation"
    model_cfg_fixture.observation_embedding = ObservationEmbeddingConfig(
        n_vars=len(OBS_VARS), d_embed=4, hidden_channels=8
    )
    pred = _make_oat_predictor(model_cfg_fixture, transform_fixture)
    batch = pred._normalize_batch(batch_fixture)
    obs_data = batch["observation_data"]
    obs_data["mask_obs"][:, 0, 1] = 0.0
    obs_data["obs"][:, 0, 1] = float("nan")
    _, _, loss, _, _ = pred.forward(batch, return_loss_and_targets=True)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in pred.model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_setup_oat_cfg_with_empty_observation_embedding_uses_defaults():
    cfg = _compose_cfg([])
    with open_dict(cfg.model):
        cfg.model.observation_embedding = {}
    oat_cfg = setup_oat_cfg(cfg, logging.getLogger(__name__))
    assert oat_cfg.observation_embedding == ObservationEmbeddingConfig(
        n_vars=len(cfg.dataset.obs_vars)
    )


def test_hydra_config_with_query_noise():
    cfg = _compose_cfg(["model/query_noise=rff"])
    qn = setup_oat_cfg(cfg, logging.getLogger(__name__)).query_noise
    assert qn == QueryNoiseConfig(
        length_scales_km=(2, 8, 32, 128, 512),
        km_per_unit=cfg.dataset.index_coord_km_per_unit,
        channels_per_scale=4,
        num_frequencies=128,
        mlp_hidden_layers=2,
        train_members=8,
        vs_weight=0.0,
        vs_p=0.5,
    )


def test_setup_oat_cfg_without_query_noise_key():
    cfg = _compose_cfg([])
    with open_dict(cfg.model):
        cfg.model.pop("query_noise", None)
    assert setup_oat_cfg(cfg, logging.getLogger(__name__)).query_noise is None


def test_predict_step_distribution_raises_with_query_noise(
    batch_fixture, transform_fixture, model_cfg_fixture
):
    model_cfg_fixture.pre_oat_loss_weight = 0.1
    model_cfg_fixture.query_noise = QueryNoiseConfig(
        length_scales_km=(2.0, 8.0),
        km_per_unit=1.0,
        channels_per_scale=2,
        num_frequencies=8,
        mlp_hidden_layers=1,
        train_members=2,
    )
    pred = _make_oat_predictor(model_cfg_fixture, transform_fixture)
    with pytest.raises(ValueError, match="query noise"):
        pred.predict_step(batch_fixture, 0, return_as="distribution")
