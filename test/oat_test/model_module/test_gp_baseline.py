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


class TestGPBaselineFeatureSlicing:
    def test_geo_in_kernel_only(self):
        # geo=0-2, nwp=3-5, time_emb skipped in both
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=True, use_topography=False)
        )
        assert 0 not in model.mean_indices
        assert 1 not in model.mean_indices
        assert 2 not in model.mean_indices
        assert model.stationary_covar_indices[:3] == (0, 1, 2)
        assert model.non_stationary_covar_indices[:3] == (0, 1, 2)

    def test_nwp_in_both_mean_and_kernel(self):
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=False, use_topography=False)
        )
        assert model.mean_indices == (3, 4, 5)
        assert model.stationary_covar_indices == (
            0,
            1,
            2,
        )
        assert model.non_stationary_covar_indices == (
            0,
            1,
            2,
            3,
            4,
            5,
        )

    def test_time_emb_excluded_from_both_by_default(self):
        # geo=0-2, nwp=3-5, time_emb=6-9 (not in groups), feature_dim=10
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=True, use_topography=False)
        )
        assert model.mean_indices == (3, 4, 5)
        assert model.stationary_covar_indices == (
            0,
            1,
            2,
        )
        assert model.non_stationary_covar_indices == (
            0,
            1,
            2,
            3,
            4,
            5,
        )
        assert model.feature_dim == 10

    def test_time_emb_included_when_in_groups(self):
        # geo=0-2, nwp=3-5, time_emb=6-9, feature_dim=10
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=False,
                mean_groups=["nwp", "time_emb"],
                stationary_cov_groups=["geo", "nwp", "time_emb"],
                non_stationary_cov_groups=["geo", "nwp", "time_emb"],
            )
        )
        assert model.mean_indices == (3, 4, 5, 6, 7, 8, 9)
        assert model.stationary_covar_indices == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        assert model.non_stationary_covar_indices == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def test_gp_modules_use_neural_mean_and_product_kernel(self):
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=True, use_topography=False)
        )
        assert type(model.gp.mean_module).__name__ == "NeuralMean"
        assert type(model.gp.covar_module).__name__ == "ProductKernel"

    def test_stationary_kernel_has_per_dimension_lengthscale(self):
        # stationary_cov_groups defaults to ["geo"] -> 3 dims, each with its own
        # learned lengthscale (ARD), so lengthscale is a length-3 vector.
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=True, use_topography=False)
        )
        stationary_kernel = model.gp.covar_module.kernels[0]
        lengthscale = stationary_kernel.kernel.base_kernel.lengthscale
        assert lengthscale.shape[-1] == len(model.stationary_covar_indices) == 3


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


class TestGPBaselineDataSpaceCRPS:
    def test_default_is_not_log(self):
        """Without set_output_normalization the loss stays plain Gaussian CRPS."""
        model = GPBaseline(make_config(loss_fn="crps"))
        assert bool(model.output_is_log) is False

    def test_set_output_normalization_registers_log_flag(self):
        model = GPBaseline(make_config(loss_fn="crps"))
        model.set_output_normalization(
            loc=torch.tensor([0.5]), scale=torch.tensor([2.0]), is_log=True
        )
        assert bool(model.output_is_log) is True
        assert torch.allclose(model.output_loc, torch.tensor([0.5]))
        assert torch.allclose(model.output_scale, torch.tensor([2.0]))

    def test_lognormal_crps_loss_matches_closed_form(self):
        """_crps_loss with is_log=True must equal LogNormalOAT.crps in data space."""
        from oat.model_module.distributions import LogNormalOAT

        model = GPBaseline(make_config(loss_fn="crps"))
        loc, scale = torch.tensor([0.3]), torch.tensor([1.7])
        model.set_output_normalization(loc=loc, scale=scale, is_log=True)

        mu = torch.tensor([0.1, -0.4, 0.8])
        sigma = torch.tensor([0.5, 0.9, 0.3])
        target = torch.tensor([-0.2, 0.6, 0.05])  # standardized-log target

        got = model._crps_loss(target, mu, sigma)
        expected = (
            LogNormalOAT(mu * scale + loc, sigma * scale)
            .crps(torch.exp(target * scale + loc))
            .mean()
        )
        assert torch.allclose(got, expected)

    def test_lognormal_crps_differs_from_logspace_crps(self):
        """The data-space loss must differ from the log-space Gaussian CRPS."""
        model = GPBaseline(make_config(loss_fn="crps"))
        mu = torch.tensor([0.1, -0.4, 0.8])
        sigma = torch.tensor([0.5, 0.9, 0.3])
        target = torch.tensor([-0.2, 0.6, 0.05])

        logspace = model._crps_loss(target, mu, sigma)  # is_log=False default
        model.set_output_normalization(
            loc=torch.tensor([0.3]), scale=torch.tensor([1.7]), is_log=True
        )
        dataspace = model._crps_loss(target, mu, sigma)
        assert not torch.allclose(logspace, dataspace)

    def test_lognormal_crps_stable_for_large_sigma(self):
        """Regression: a large data-space sigma (sigma * scale) must not overflow
        to NaN. exp(mu + sigma^2/2) overflows fp32 and m*Phi -> inf*0 = NaN; the
        log_ndtr formulation keeps the loss and gradients finite."""
        model = GPBaseline(make_config(loss_fn="crps"))
        model.set_output_normalization(
            loc=torch.tensor([0.0]), scale=torch.tensor([5.0]), is_log=True
        )
        mu = torch.tensor([0.5, -1.0], requires_grad=True)
        sigma = torch.tensor([3.0, 2.5], requires_grad=True)  # sigma_log = 15.0, 12.5
        target = torch.tensor([0.2, -0.3])
        loss = model._crps_loss(target, mu, sigma)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(sigma.grad).all()

    def test_lognormal_crps_stable_for_zero_sigma(self):
        """Regression: a collapsed scale (sigma -> 0) must not divide-by-zero."""
        model = GPBaseline(make_config(loss_fn="crps"))
        model.set_output_normalization(
            loc=torch.tensor([0.0]), scale=torch.tensor([1.0]), is_log=True
        )
        loss = model._crps_loss(
            torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([0.0])
        )
        assert torch.isfinite(loss)

    def test_lognormal_crps_loss_is_finite_with_zero_target(self):
        """Standardized target maps to a strictly positive data-space value, so a
        raw zero (offset only) stays finite and differentiable."""
        model = GPBaseline(make_config(loss_fn="crps"))
        model.set_output_normalization(
            loc=torch.tensor([0.0]), scale=torch.tensor([1.0]), is_log=True
        )
        mu = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([0.5])
        # log(0 + offset) with offset=1e-3 -> ~-6.9 in raw-log space
        target = torch.tensor([-6.9])
        loss = model._crps_loss(target, mu, sigma)
        assert torch.isfinite(loss)
        loss.backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()


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
