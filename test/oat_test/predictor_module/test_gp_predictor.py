import pytest
import torch

from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline
from oat.predictor_module.gp_predictor import GPPredictor

NWP_VARS = ["T_2M", "U_10M", "V_10M"]
OBS_VAR = ["temperature"]


class IdentityTransform:
    def __call__(self, batch):
        return batch

    def inverse(self, x):
        return x


def make_predictor(loss_fn="crps") -> GPPredictor:
    cfg = GPBaselineConfig(
        obs_vars=OBS_VAR,
        nwp_vars=NWP_VARS,
        loss_fn=loss_fn,
        num_samples=3,
    )
    model = GPBaseline(cfg)
    return GPPredictor(
        gp_model=model,
        input_transform=IdentityTransform(),
        output_transform=IdentityTransform(),
        output_var_names=OBS_VAR,
    )


def make_batch(B=2, N_obs=8, N_q=5, n_nwp=3, n_topo=5):
    return {
        "global_data": {"time_emb": torch.randn(B, 4)},
        "observation_data": {
            "geo": torch.randn(B, N_obs, 3),
            "global_tokens": torch.randn(B, N_obs, n_nwp),
            "topo": torch.randn(B, N_obs, n_topo),
            "obs": torch.randn(B, N_obs, 1),
            "mask_obs": torch.ones(B, N_obs, 1),
        },
        "query_data": {
            "geo": torch.randn(B, N_q, 3),
            "global_tokens": torch.randn(B, N_q, n_nwp),
            "topo": torch.randn(B, N_q, n_topo),
            "obs": torch.randn(B, N_q, 1),
            "mask_obs": torch.ones(B, N_q, 1),
        },
    }


class TestGPPredictorInit:
    def test_instantiation(self):
        p = make_predictor()
        assert hasattr(p, "model")
        assert hasattr(p, "input_transform")
        assert hasattr(p, "output_transform")

    def test_var_index_mapping(self):
        p = make_predictor()
        assert p.var_to_indx["temperature"] == 0
        assert p.indx_to_var[0] == "temperature"

    def test_metrics_registered(self):
        p = make_predictor()
        assert "point_mae" in p.metrics_fn
        assert "crps" in p.metrics_fn


class TestGPPredictorTrainingStep:
    @pytest.mark.parametrize("loss_fn", ["crps", "mll"])
    def test_returns_scalar_finite_loss(self, loss_fn):
        p = make_predictor(loss_fn=loss_fn)
        loss = p.training_step(make_batch(), batch_idx=0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("loss_fn", ["crps", "mll"])
    def test_loss_is_differentiable(self, loss_fn):
        p = make_predictor(loss_fn=loss_fn)
        loss = p.training_step(make_batch(), batch_idx=0)
        loss.backward()
        has_grad = any(
            param.grad is not None
            for param in p.model.parameters()
            if param.requires_grad
        )
        assert has_grad


class TestGPPredictorValidationStep:
    def test_returns_finite_loss(self):
        p = make_predictor()
        loss = p.validation_step(make_batch(), batch_idx=0)
        assert torch.isfinite(loss)


class TestGPPredictorConfigureOptimizers:
    def test_returns_optimizer(self):
        p = make_predictor()
        result = p.configure_optimizers()
        if isinstance(result, dict):
            assert "optimizer" in result
        else:
            assert hasattr(result, "step")


class TestGPPredictorPredictStep:
    def test_predict_step_returns_samples_shape(self):
        p = make_predictor()
        batch = make_batch(B=1, N_q=5)
        out = p.predict_step(batch, batch_idx=0)
        # default: [B, N_q, num_samples, n_vars]
        assert out.shape == (1, 5, 3, 1)

    def test_predict_step_n_samples_override(self):
        p = make_predictor()
        batch = make_batch(B=1, N_q=5)
        out = p.predict_step(batch, batch_idx=0, n_samples=7)
        assert out.shape == (1, 5, 7, 1)

    def test_predict_step_distribution_raises(self):
        p = make_predictor()
        batch = make_batch(B=1, N_q=5)
        with pytest.raises(NotImplementedError):
            p.predict_step(batch, batch_idx=0, return_as="distribution")


@pytest.mark.parametrize(
    "model_name,deep", [("gp_oat_baseline", False), ("deep_gp_oat_baseline", True)]
)
def test_gp_oat_baseline_config_builds_trunk_backed_gp(model_name, deep):
    """The gp_oat configs must compose into a GP whose mean is the uninformed
    OAT trunk, with no output head. The deep variant additionally feeds those
    same tokens to the non-stationary kernel."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from oat.predictor_module.gp_predictor import setup_gp_predictor

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "predictor=gp_predictor",
                "dataset=cosmo_peakweather/temp",
            ],
        )

    predictor = setup_gp_predictor(
        cfg,
        list(cfg.dataset.obs_vars),
        (IdentityTransform(), IdentityTransform()),
    )

    gp = predictor.model
    oat_indices = tuple(range(3, 3 + cfg.model.d_model))
    assert gp.pre_oat is not None
    assert gp.pre_oat.output_head is None
    # geo (3) + fusion tokens (d_model); no hand-crafted nwp/time/topo groups.
    assert gp.feature_dim == 3 + cfg.model.d_model
    assert gp.mean_indices == oat_indices
    assert gp.stationary_covar_indices == (0, 1, 2)
    assert gp.non_stationary_covar_indices == (oat_indices if deep else ())
    assert type(gp.gp.covar_module).__name__ == (
        "ProductKernel" if deep else "ScaledRBFKernel"
    )


@pytest.mark.parametrize("model_name", ["gp_oat_baseline", "deep_gp_oat_baseline"])
def test_gp_oat_mean_matches_uninformed_baseline_trunk(model_name):
    """The gp_oat mean function must be the same architecture as the
    uninformed OAT baseline it is compared against."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from oat.predictor_module.oat_predictor import setup_backbone_cfgs

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")

    def backbone_for(name):
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"model={name}",
                    "dataset=cosmo_peakweather/temp",
                ],
            )
        return setup_backbone_cfgs(cfg)

    assert backbone_for(model_name) == backbone_for("uninformed_oat")


@pytest.mark.parametrize(
    "model_name,is_gp",
    [
        ("gp_oat_baseline", True),
        ("deep_gp_oat_baseline", True),
        ("gp_baseline", True),
        ("deep_gp_baseline", True),
        ("oat", False),
        ("uninformed_oat", False),
    ],
)
def test_train_model_routes_gp_configs_to_the_gp_path(model_name, is_gp):
    """Every GPBaseline config must take the GP training branch and select
    checkpoints on the data-space CRPS, not on the log-space val_loss."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from scripts.train_model import _is_gp_model, _resolve_checkpoint_monitor

    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"model={model_name}",
                "dataset=cosmo_peakweather/legacy/precip",
            ],
        )

    assert _is_gp_model(cfg) is is_gp
    expected = ("val_crps_precipitation", "min") if is_gp else ("val_loss", "min")
    assert _resolve_checkpoint_monitor(cfg) == expected
