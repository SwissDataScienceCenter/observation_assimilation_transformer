import pytest
import torch

from oat.data_module.transform import (
    ComposedTransform,
    LogTransform,
    StandardNormalizeTransform,
)
from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline
from oat.predictor_module.gp_predictor import GPPredictor, extract_output_normalization


FIELD_EXTENT = {"min": [2255500.0, 840500.0], "max": [2964500.0, 1479500.0]}
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
        field_extent=FIELD_EXTENT,
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


def make_batch(B=2, N_obs=8, N_q=5, n_nwp=3, n_topo=5, H=32, W=32):
    obs_coords = torch.rand(B, N_obs, 2) * 700000 + 2255500.0
    query_coords = torch.rand(B, N_q, 2) * 700000 + 2255500.0
    return {
        "global_data": {
            "global_field": torch.randn(B, n_nwp + n_topo, H, W),
            "time_emb": torch.randn(B, 4),
        },
        "observation_data": {
            "coords": obs_coords,
            "geo": torch.randn(B, N_obs, 3),
            "topo": torch.randn(B, N_obs, n_topo, 32, 32),
            "obs": torch.randn(B, N_obs, 1),
            "mask_obs": torch.ones(B, N_obs, 1),
        },
        "query_data": {
            "coords": query_coords,
            "geo": torch.randn(B, N_q, 3),
            "topo": torch.randn(B, N_q, n_topo, 32, 32),
            "obs": torch.randn(B, N_q, 1),
            "mask_obs": torch.ones(B, N_q, 1),
        },
    }


class TestExtractOutputNormalization:
    def test_log_plus_standardize(self):
        """Precip-style: log then standardize -> is_log True, loc/scale recovered."""
        mean, std = torch.tensor([1.5]), torch.tensor([0.7])
        ot = ComposedTransform(
            [
                LogTransform(offset=1e-3, apply_to=[0]),
                StandardNormalizeTransform(mean=mean, std=std),
            ]
        )
        loc, scale, is_log = extract_output_normalization(ot, var_index=0)
        assert is_log is True
        # z_log = z_std * std + mean  ->  loc = mean, scale = std
        assert torch.allclose(loc, mean)
        assert torch.allclose(scale, std)

    def test_standardize_only_is_not_log(self):
        """Temperature-style: log applies to no index -> is_log False."""
        mean, std = torch.tensor([2.0]), torch.tensor([3.0])
        ot = ComposedTransform(
            [
                LogTransform(offset=1e-3, apply_to=[]),
                StandardNormalizeTransform(mean=mean, std=std),
            ]
        )
        loc, scale, is_log = extract_output_normalization(ot, var_index=0)
        assert is_log is False
        assert torch.allclose(loc, mean)
        assert torch.allclose(scale, std)

    def test_identity_transform_defaults(self):
        """A transform without recognizable components -> safe defaults."""
        loc, scale, is_log = extract_output_normalization(IdentityTransform())
        assert is_log is False
        assert torch.allclose(loc, torch.zeros(1))
        assert torch.allclose(scale, torch.ones(1))


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
