import pytest
import torch

from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline
from oat.predictor_module.gp_predictor import GPPredictor


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
