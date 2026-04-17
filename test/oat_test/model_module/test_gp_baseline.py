import pytest
import torch

from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline


FIELD_EXTENT = {"min": [2255500.0, 840500.0], "max": [2964500.0, 1479500.0]}
NWP_VARS = ["T_2M", "U_10M", "V_10M"]


def make_config(**kwargs) -> GPBaselineConfig:
    defaults = dict(
        obs_vars=["temperature"],
        nwp_vars=NWP_VARS,
        field_extent=FIELD_EXTENT,
    )
    defaults.update(kwargs)
    return GPBaselineConfig(**defaults)


def make_batch(B=2, N_obs=10, N_q=8, n_nwp=3, n_topo=5, H=32, W=32):
    """Synthetic batch matching CosmoPeakWeather format (one obs var)."""
    obs_coords = torch.rand(B, N_obs, 2) * 700000 + 2255500.0
    query_coords = torch.rand(B, N_q, 2) * 700000 + 2255500.0
    global_data = {
        "global_field": torch.randn(B, n_nwp + n_topo, H, W),
        "time_emb": torch.randn(B, 4),
    }
    observation_data = {
        "coords": obs_coords,
        "geo": torch.randn(B, N_obs, 3),
        "topo": torch.randn(B, N_obs, n_topo, 32, 32),
        "obs": torch.randn(B, N_obs, 1),
        "mask_obs": torch.ones(B, N_obs, 1),
    }
    query_data = {
        "coords": query_coords,
        "geo": torch.randn(B, N_q, 3),
        "topo": torch.randn(B, N_q, n_topo, 32, 32),
        "obs": torch.randn(B, N_q, 1),
        "mask_obs": torch.ones(B, N_q, 1),
    }
    return global_data, observation_data, query_data


class TestGPBaselineConfigValidation:
    def test_multivariate_raises(self):
        """GPBaseline must reject more than one obs variable."""
        with pytest.raises(ValueError, match="univariate"):
            GPBaseline(make_config(obs_vars=["temperature", "wind_gust"]))

    def test_single_variable_ok(self):
        model = GPBaseline(make_config(obs_vars=["temperature"]))
        assert model is not None

    def test_invalid_loss_fn_raises(self):
        with pytest.raises(ValueError, match="loss_fn"):
            GPBaseline(make_config(loss_fn="unknown"))

    def test_valid_loss_fn_mll(self):
        model = GPBaseline(make_config(loss_fn="mll"))
        assert model is not None

    def test_valid_loss_fn_crps(self):
        model = GPBaseline(make_config(loss_fn="crps"))
        assert model is not None


class TestGPBaselineFeatureDim:
    def test_feature_dim_full(self):
        model = GPBaseline(make_config(use_nwp=True, use_time_emb=True))
        assert model.feature_dim == 10

    def test_feature_dim_geo_nwp_time_topography(self):
        topo_cfg = {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 3,
            "initial_hw": (32, 32),
            "kernel_sizes": 3,
            "act_fn": "ReLU",
            "use_batchnorm": True,
        }
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=True,
                topography_encoder_kwargs=topo_cfg,
            )
        )
        assert model.feature_dim == 3 + 3 + 4 + 16

    def test_feature_dim_geo_only(self):
        model = GPBaseline(make_config(use_nwp=False, use_time_emb=False))
        assert model.feature_dim == 3

    def test_feature_dim_geo_plus_nwp(self):
        model = GPBaseline(make_config(use_nwp=True, use_time_emb=False))
        assert model.feature_dim == 6

    def test_feature_dim_without_topography_matches_previous_behavior(self):
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=False,
            )
        )
        assert model.feature_dim == 10


class TestGPBaselineFeatureExtraction:
    def test_shape_full_features(self):
        model = GPBaseline(make_config(use_nwp=True, use_time_emb=True))
        B, N_obs = 2, 10
        global_data, obs_data, _ = make_batch(B=B, N_obs=N_obs)
        feats = model._extract_features(global_data, obs_data)
        assert feats.shape == (B, N_obs, 10)

    def test_shape_includes_topography_embedding(self):
        topo_cfg = {
            "in_channels": 5,
            "out_channels": 16,
            "n_layers": 3,
            "initial_hw": (32, 32),
            "kernel_sizes": 3,
            "act_fn": "ReLU",
            "use_batchnorm": True,
        }
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=True,
                topography_encoder_kwargs=topo_cfg,
            )
        )
        B, N_obs = 2, 10
        global_data, obs_data, _ = make_batch(B=B, N_obs=N_obs, n_topo=5)
        feats = model._extract_features(global_data, obs_data)
        assert feats.shape == (B, N_obs, 3 + 3 + 4 + 16)

    def test_shape_geo_only(self):
        model = GPBaseline(make_config(use_nwp=False, use_time_emb=False))
        B, N_obs = 2, 10
        global_data, obs_data, _ = make_batch(B=B, N_obs=N_obs)
        feats = model._extract_features(global_data, obs_data)
        assert feats.shape == (B, N_obs, 3)

    def test_query_features_have_same_dim_as_obs_features(self):
        model = GPBaseline(make_config())
        B = 2
        global_data, obs_data, query_data = make_batch(B=B, N_obs=10, N_q=6)
        obs_feats = model._extract_features(global_data, obs_data)
        qry_feats = model._extract_features(global_data, query_data)
        assert obs_feats.shape[-1] == qry_feats.shape[-1]


class TestGPBaselineForward:
    def test_forward_output_shape(self):
        model = GPBaseline(make_config(num_samples=5))
        B, N_obs, N_q = 2, 10, 6
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        out = model.forward(global_data, obs_data, query_data, num_samples=5)
        assert "samples" in out
        assert out["samples"].shape == (B, N_q, 5, 1)

    def test_requires_observation_flag(self):
        model = GPBaseline(make_config())
        assert model.requires_observation is True


class TestGPBaselineForwardWithLossCRPS:
    def test_returns_samples_and_scalar_loss(self):
        model = GPBaseline(make_config(loss_fn="crps", num_samples=3))
        B, N_obs, N_q = 2, 10, 6
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        target_obs = query_data["obs"]
        target_mask = query_data["mask_obs"]
        out, loss = model.forward_with_loss(
            global_data, obs_data, query_data, target_obs, target_mask, num_samples=3
        )
        assert "samples" in out
        assert out["samples"].shape == (B, N_q, 3, 1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_has_gradients(self):
        model = GPBaseline(make_config(loss_fn="crps", num_samples=3))
        B, N_obs, N_q = 1, 8, 4
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        target_obs = query_data["obs"]
        target_mask = query_data["mask_obs"]
        _, loss = model.forward_with_loss(
            global_data, obs_data, query_data, target_obs, target_mask, num_samples=3
        )
        loss.backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad

    def test_partial_context_mask(self):
        model = GPBaseline(make_config(loss_fn="crps", num_samples=3))
        B, N_obs, N_q = 1, 10, 5
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        obs_data["mask_obs"][:, :5, :] = 0.0
        target_obs = query_data["obs"]
        target_mask = query_data["mask_obs"]
        _, loss = model.forward_with_loss(
            global_data, obs_data, query_data, target_obs, target_mask, num_samples=3
        )
        assert torch.isfinite(loss)


class TestGPBaselineForwardWithLossMLL:
    def test_returns_samples_and_scalar_loss(self):
        model = GPBaseline(make_config(loss_fn="mll", num_samples=3))
        B, N_obs, N_q = 2, 10, 6
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        target_obs = query_data["obs"]
        target_mask = query_data["mask_obs"]
        out, loss = model.forward_with_loss(
            global_data, obs_data, query_data, target_obs, target_mask, num_samples=3
        )
        assert "samples" in out
        assert out["samples"].shape == (B, N_q, 3, 1)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_loss_has_gradients(self):
        model = GPBaseline(make_config(loss_fn="mll", num_samples=3))
        B, N_obs, N_q = 1, 8, 4
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        target_obs = query_data["obs"]
        target_mask = query_data["mask_obs"]
        _, loss = model.forward_with_loss(
            global_data, obs_data, query_data, target_obs, target_mask, num_samples=3
        )
        loss.backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad
