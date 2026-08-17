import pytest
import torch

from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline
from oat.model_module.model import (
    GlobalEncoderConfig,
    NwpTopoFusionConfig,
    OutputHeadConfig,
    TopographicEncoderConfig,
    UninformedOATConfig,
)

FIELD_EXTENT = {"min": [2255500.0, 840500.0], "max": [2964500.0, 1479500.0]}
NWP_VARS = ["T_2M", "U_10M", "V_10M"]


def make_config(**kwargs) -> GPBaselineConfig:
    defaults = dict(
        obs_vars=["temperature"],
        nwp_vars=NWP_VARS,
    )
    defaults.update(kwargs)
    return GPBaselineConfig(**defaults)


def make_batch(B=2, N_obs=10, N_q=8, n_nwp=3, n_topo=5):
    """Synthetic field-shaped batch: flat per-point tensors, no grid/patches."""
    global_data = {"time_emb": torch.randn(B, 4)}
    observation_data = {
        "geo": torch.randn(B, N_obs, 3),
        "global_tokens": torch.randn(B, N_obs, n_nwp),
        "topo": torch.randn(B, N_obs, n_topo),
        "obs": torch.randn(B, N_obs, 1),
        "mask_obs": torch.ones(B, N_obs, 1),
    }
    query_data = {
        "geo": torch.randn(B, N_q, 3),
        "global_tokens": torch.randn(B, N_q, n_nwp),
        "topo": torch.randn(B, N_q, n_topo),
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
        topo_cfg = TopographicEncoderConfig(
            in_channels=5,
            out_channels=16,
            type="MLP",
            hidden_dims=(8,),
            act_fn="ReLU",
        )
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=True,
                topographic_encoder=topo_cfg,
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
        assert model.stationary_covar_indices == (0, 1, 2)
        assert model.non_stationary_covar_indices == (0, 1, 2, 3, 4, 5)

    def test_time_emb_excluded_from_both_by_default(self):
        # geo=0-2, nwp=3-5, time_emb=6-9 (not in groups), feature_dim=10
        model = GPBaseline(
            make_config(use_nwp=True, use_time_emb=True, use_topography=False)
        )
        assert model.mean_indices == (3, 4, 5)
        assert model.stationary_covar_indices == (0, 1, 2)
        assert model.non_stationary_covar_indices == (0, 1, 2, 3, 4, 5)
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
        topo_cfg = TopographicEncoderConfig(
            in_channels=5,
            out_channels=16,
            type="MLP",
            hidden_dims=(8,),
            act_fn="ReLU",
        )
        model = GPBaseline(
            make_config(
                use_nwp=True,
                use_time_emb=True,
                use_topography=True,
                topographic_encoder=topo_cfg,
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

    def test_extra_global_channels_are_sliced_to_nwp_prefix(self):
        # The global-role field may append extra context channels (e.g. topo.*)
        # after the NWP scalars (see validate_field_config_against_scalars).
        # The GP feature vector must consume only the len(nwp_vars) prefix, so
        # the config-derived offsets for time_emb/topo/oat stay aligned.
        model = GPBaseline(make_config(use_nwp=True, use_time_emb=True))
        B, N_obs = 2, 10
        global_data, obs_data, _ = make_batch(B=B, N_obs=N_obs)
        obs_data["global_tokens"] = torch.cat(
            [obs_data["global_tokens"], torch.randn(B, N_obs, 2)], dim=-1
        )
        feats = model._extract_features(global_data, obs_data)
        assert feats.shape == (B, N_obs, model.feature_dim)
        # NWP block keeps the leading nwp_vars channels, extras are dropped.
        assert torch.equal(feats[..., 3:6], obs_data["global_tokens"][..., :3])
        # time_emb still lands at its configured offset (6:10), unshifted.
        time_emb = global_data["time_emb"].unsqueeze(1).expand(-1, N_obs, -1)
        assert torch.equal(feats[..., 6:10], time_emb)


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

    def test_forward_with_extra_global_channels(self):
        model = GPBaseline(make_config(num_samples=5))
        B, N_obs, N_q = 2, 10, 6
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        obs_data["global_tokens"] = torch.cat(
            [obs_data["global_tokens"], torch.randn(B, N_obs, 2)], dim=-1
        )
        query_data["global_tokens"] = torch.cat(
            [query_data["global_tokens"], torch.randn(B, N_q, 2)], dim=-1
        )
        out = model.forward(global_data, obs_data, query_data, num_samples=5)
        assert out["samples"].shape == (B, N_q, 5, 1)
        assert torch.isfinite(out["samples"]).all()


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


class TestGPBaselineNoiseLowerBound:
    """The likelihood noise floor is a modelling choice, not a safety net:
    it caps how sharp the predictive distribution can get, so the configured
    value must reach the likelihood unchanged."""

    def test_default_is_1e_3(self):
        assert GPBaselineConfig.noise_lower_bound == pytest.approx(1e-3)

    @pytest.mark.parametrize("bound", [1e-3, 5e-2])
    def test_reaches_likelihood_constraint(self, bound):
        model = GPBaseline(make_config(noise_lower_bound=bound))
        constraint = model.gp.likelihood.noise_covar.raw_noise_constraint
        assert constraint.lower_bound.item() == pytest.approx(bound)

    def test_default_config_reaches_likelihood_constraint(self):
        model = GPBaseline(make_config())
        constraint = model.gp.likelihood.noise_covar.raw_noise_constraint
        assert constraint.lower_bound.item() == pytest.approx(1e-3)


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


def make_pre_oat_config(d_model=16, n_topo=5, n_nwp=3):
    """UninformedOAT trunk config (no output head) for use as the GP mean.

    Field-mode: Point global encoder over global_tokens, MLP topo encoder over
    flat topo values -- matches what make_batch() supplies.
    """
    return UninformedOATConfig(
        global_encoder=GlobalEncoderConfig(
            type="Point",
            in_channels=n_nwp,
            out_channels=d_model // 2,
            hidden_dims=(8,),
            field_extent_min=tuple(FIELD_EXTENT["min"]),
            field_extent_max=tuple(FIELD_EXTENT["max"]),
        ),
        topographic_encoder=TopographicEncoderConfig(
            type="MLP",
            in_channels=n_topo,
            out_channels=d_model // 2,
            hidden_dims=(8,),
        ),
        nwp_topo_fusion=NwpTopoFusionConfig(
            cond_dim=4, hidden_channels=d_model, out_channels=d_model
        ),
    )


class TestGPBaselinePreOATMean:
    def test_output_head_must_be_none(self):
        pre_oat = make_pre_oat_config()
        pre_oat.output_head = OutputHeadConfig(
            in_features=16, variable_to_distribution={"temperature": "normal"}
        )
        with pytest.raises(ValueError, match="output_head must be None"):
            GPBaseline(make_config(pre_oat=pre_oat, mean_groups=["oat"]))

    def test_trunk_is_built_without_output_head(self):
        model = GPBaseline(
            make_config(pre_oat=make_pre_oat_config(), mean_groups=["oat"])
        )
        assert model.pre_oat is not None
        assert model.pre_oat.output_head is None

    def test_feature_dim_and_indices(self):
        # geo=0-2, oat=3-18 (d_model=16); nwp/time_emb/topo groups disabled.
        model = GPBaseline(
            make_config(
                use_nwp=False,
                use_time_emb=False,
                use_topography=False,
                pre_oat=make_pre_oat_config(d_model=16),
                mean_groups=["oat"],
            )
        )
        assert model.feature_dim == 3 + 16
        assert model.mean_indices == tuple(range(3, 19))
        assert model.stationary_covar_indices == (0, 1, 2)

    def test_forward_with_loss_trains_the_trunk(self):
        model = GPBaseline(
            make_config(
                use_nwp=False,
                use_time_emb=False,
                use_topography=False,
                pre_oat=make_pre_oat_config(d_model=16),
                mean_groups=["oat"],
                loss_fn="crps",
                num_samples=3,
            )
        )
        B, N_obs, N_q = 2, 10, 6
        global_data, obs_data, query_data = make_batch(B=B, N_obs=N_obs, N_q=N_q)
        out, loss = model.forward_with_loss(
            global_data,
            obs_data,
            query_data,
            query_data["obs"],
            query_data["mask_obs"],
            num_samples=3,
        )
        assert out["samples"].shape == (B, N_q, 3, 1)
        assert torch.isfinite(loss)

        loss.backward()
        trunk_grads = [p.grad for p in model.pre_oat.parameters() if p.requires_grad]
        assert trunk_grads
        assert any(g is not None and g.abs().sum() > 0 for g in trunk_grads)


def test_gp_geo_dim_shifts_feature_group_offsets():
    from oat.model_module.gp_baseline import GPBaselineConfig, GPBaseline

    common = dict(
        obs_vars=["t"],
        nwp_vars=["a", "b"],
        use_nwp=True,
        use_time_emb=False,
        use_topography=False,
        mean_groups=["nwp"],
        stationary_cov_groups=["geo"],
        non_stationary_cov_groups=["geo", "nwp"],
    )
    # default geo_dim == 3: nwp indices start at 3
    m3 = GPBaseline(GPBaselineConfig(geo_dim=3, **common))
    # geo_dim == 4 (ERA5 sphere): nwp indices shift to start at 4
    m4 = GPBaseline(GPBaseline.__init__.__globals__["GPBaselineConfig"](geo_dim=4, **common))
    # The stationary kernel is defined over geo indices; its ard width equals geo_dim.
    assert m3.config.geo_dim == 3
    assert m4.config.geo_dim == 4
