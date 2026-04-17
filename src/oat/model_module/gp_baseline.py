from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gpytorch
import torch
import torch.nn as nn
from obsweatherscale.kernels.neural_kernel import NeuralKernel
from obsweatherscale.kernels.scaled_rbf_kernel import ScaledRBFKernel
from obsweatherscale.means.neural_mean import NeuralMean
from obsweatherscale.models.gp_model import GPModel
from obsweatherscale.models.mlp import MLP
# from obsweatherscale.training.losses import crps_normal
import torch.distributions as dist

from oat.model_module.baseline import NWPInterpolator, NWPInterpolatorConfig
from oat.model_module.topographic_encoder import CNNTopographyEncoder


_VALID_LOSS_FNS = ("mll", "crps")

def crps_normal(
    obs: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Wrapper to compute the Continuous Ranked Probability Score (CRPS)
    for a Normal distribution.

    Parameters
    ----------
    obs : torch.Tensor
        Observed values.
    mu : torch.Tensor
        Mean of the Normal distribution.
    sigma : torch.Tensor
        Standard deviation of the Normal distribution.

    Returns
    -------
    torch.Tensor
        The CRPS for the normal distribution, averaged across all
        observations.

    Notes
    -----
    The CRPS is a proper scoring rule that measures the compatibility
    between a probability distribution and an observation. It's defined
    as the integrated squared difference between the CDF of the forecast
    distribution and the empirical CDF of the observation.

    For a normal distribution, the CRPS has the closed form:
    CRPS(N(μ, σ), y) = 
        σ * [y_norm * (2*Φ(y_norm) - 1) + 2*φ(y_norm) - 1/√π]
    where y_norm = (y - μ)/σ, Φ is the CDF and φ is the PDF of the
    standard normal.
    """
    # Create normal distribution
    normal_dist = dist.Normal(mu, sigma)

    # Compute cumul distr function and prob density function
    cdf_obs = normal_dist.cdf(obs)
    pdf_obs = normal_dist.log_prob(obs).exp()

    # Compute CRPS
    term1 = (obs - mu) * (2 * cdf_obs - 1)
    term2 = sigma * (2 * pdf_obs * sigma - 1 / torch.sqrt(torch.tensor(torch.pi)))
    crps = term1 + term2
    # normal = dist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    # scaled = (obs - mu) / sigma

    # Phi = normal.cdf(scaled)
    # phi = torch.exp(normal.log_prob(scaled))
    # crps = sigma * (scaled * (2 * Phi - 1) + 2 * phi - (1 / torch.sqrt(torch.tensor(torch.pi, device=sigma.device))))

    return crps.mean()


@dataclass
class GPBaselineConfig:
    obs_vars: List[str]
    nwp_vars: List[str]
    field_extent: Dict[str, Any]
    use_nwp: bool = True
    use_time_emb: bool = True
    use_topography: bool = False
    topography_encoder_kwargs: Optional[Dict[str, Any]] = None
    mean_hidden_dims: Tuple[int, ...] = (32, 32)
    kernel_hidden_dims: Tuple[int, ...] = (32, 32)
    kernel_embed_dim: int = 4
    noise_variance: float = 1.0
    loss_fn: str = "crps"
    num_samples: int = 100


class GPBaseline(nn.Module):
    requires_observation: bool = True

    def __init__(self, config: GPBaselineConfig) -> None:
        super().__init__()

        if len(config.obs_vars) != 1:
            raise ValueError(
                "GPBaseline is univariate: obs_vars must contain exactly one "
                f"variable, got {config.obs_vars}."
            )
        if config.loss_fn not in _VALID_LOSS_FNS:
            raise ValueError(
                f"loss_fn must be one of {_VALID_LOSS_FNS}, got '{config.loss_fn}'."
            )

        self.config = config
        self.feature_dim = 3
        if config.use_nwp:
            self.feature_dim += len(config.nwp_vars)
        if config.use_time_emb:
            self.feature_dim += 4
        if config.use_topography:
            if config.topography_encoder_kwargs is None:
                raise ValueError(
                    "topography_encoder_kwargs must be provided when "
                    "use_topography=True."
                )
            self.topography_encoder: Optional[CNNTopographyEncoder] = (
                CNNTopographyEncoder(**config.topography_encoder_kwargs)
            )
            self.feature_dim += int(config.topography_encoder_kwargs["out_channels"])
        else:
            self.topography_encoder = None

        if config.use_nwp:
            self.nwp_interpolator: Optional[NWPInterpolator] = NWPInterpolator(
                NWPInterpolatorConfig(field_extent=config.field_extent)
            )
        else:
            self.nwp_interpolator = None

        feature_dim = self.feature_dim
        mean_net = MLP(dimensions=[feature_dim, *config.mean_hidden_dims, 1])
        kernel_net = MLP(
            dimensions=[feature_dim, *config.kernel_hidden_dims, config.kernel_embed_dim]
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with torch.no_grad():
            likelihood.noise = torch.tensor(float(config.noise_variance))

        dummy_x = torch.zeros(2, feature_dim)
        dummy_y = torch.zeros(2)
        self.gp = GPModel(
            train_x=dummy_x,
            train_y=dummy_y,
            likelihood=likelihood,
            mean_module=NeuralMean(net=mean_net),
            covar_module=NeuralKernel(net=kernel_net, kernel=ScaledRBFKernel()),
        )
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

    def _extract_features(
        self,
        global_data: Dict[str, torch.Tensor],
        local_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = [local_data["geo"]]

        if self.nwp_interpolator is not None:
            n_nwp = len(self.config.nwp_vars)
            nwp_global_data = {
                **global_data,
                "global_field": global_data["global_field"][:, :n_nwp],
            }
            nwp_feats = self.nwp_interpolator(nwp_global_data, local_data)
            parts.append(nwp_feats)

        if self.config.use_time_emb:
            num_points = local_data["geo"].shape[1]
            time_emb = global_data["time_emb"].unsqueeze(1).expand(-1, num_points, -1)
            parts.append(time_emb)

        if self.topography_encoder is not None:
            topo_feats = self.topography_encoder(local_data["topo"])
            parts.append(topo_feats)

        return torch.cat(parts, dim=-1)

    def _posterior_predictive(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        tx: torch.Tensor,
    ) -> gpytorch.distributions.base_distributions.Normal:
        self.gp.set_train_data(inputs=cx, targets=cy, strict=False)
        self.gp.eval()
        self.gp.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            pred = self.gp(tx)
        return self.gp.likelihood(pred)

    def _mll_loss_for_context(self, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        self.gp.set_train_data(inputs=cx, targets=cy, strict=False)
        self.gp.train()
        self.gp.likelihood.train()
        output = self.gp(cx)
        return -self._mll(output, cy)

    @staticmethod
    def _ensure_non_empty_context(
        cx: torch.Tensor,
        cy: torch.Tensor,
        tx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cx.numel() > 0:
            return cx, cy
        fallback_x = tx[:1].detach().clone()
        fallback_y = torch.zeros(1, device=tx.device, dtype=tx.dtype)
        return fallback_x, fallback_y

    def forward(
        self,
        global_data: Dict[str, torch.Tensor],
        observation_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if num_samples is None:
            num_samples = self.config.num_samples

        context_x = self._extract_features(global_data, observation_data)
        target_x = self._extract_features(global_data, query_data)
        context_y = observation_data["obs"][..., 0]
        context_mask = observation_data["mask_obs"][..., 0]

        batch_size = target_x.shape[0]
        var_samples: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            valid = context_mask[batch_idx].bool()
            cx = context_x[batch_idx][valid]
            cy = context_y[batch_idx][valid]
            tx = target_x[batch_idx]
            cx, cy = self._ensure_non_empty_context(cx, cy, tx)

            with torch.no_grad():
                lik_pred = self._posterior_predictive(cx, cy, tx)
                sample_values = lik_pred.rsample(torch.Size([num_samples]))
            var_samples.append(sample_values.T)

        return {"samples": torch.stack(var_samples, dim=0).unsqueeze(-1)}

    def forward_with_loss(
        self,
        global_data: Dict[str, torch.Tensor],
        observation_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        target_obs: torch.Tensor,
        target_mask_obs: torch.Tensor,
        num_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if num_samples is None:
            num_samples = self.config.num_samples

        context_x = self._extract_features(global_data, observation_data)
        target_x = self._extract_features(global_data, query_data)
        context_y = observation_data["obs"][..., 0]
        context_mask = observation_data["mask_obs"][..., 0]
        target_y = target_obs[..., 0]
        target_mask = target_mask_obs[..., 0]

        batch_size = context_x.shape[0]
        var_samples: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=context_x.device)
        n_terms = 0

        for batch_idx in range(batch_size):
            valid_ctx = context_mask[batch_idx].bool()
            cx = context_x[batch_idx][valid_ctx]
            cy = context_y[batch_idx][valid_ctx]
            tx = target_x[batch_idx]
            ty = target_y[batch_idx]
            valid_tgt = target_mask[batch_idx].bool()
            cx, cy = self._ensure_non_empty_context(cx, cy, tx)

            if self.config.loss_fn == "mll":
                total_loss = total_loss + self._mll_loss_for_context(cx, cy)
                n_terms += 1
                with torch.no_grad():
                    lik_pred = self._posterior_predictive(cx, cy, tx)
                    sample_values = lik_pred.rsample(torch.Size([num_samples]))
            else:
                lik_pred = self._posterior_predictive(cx, cy, tx)
                mu = lik_pred.mean
                sigma = lik_pred.stddev

                if valid_tgt.any():
                    crps = crps_normal(ty[valid_tgt], mu[valid_tgt], sigma[valid_tgt])
                    total_loss = total_loss + crps.mean()
                    n_terms += 1

                with torch.no_grad():
                    sample_values = lik_pred.rsample(torch.Size([num_samples]))

            var_samples.append(sample_values.T)

        samples = torch.stack(var_samples, dim=0).unsqueeze(-1)
        loss = total_loss / max(n_terms, 1)
        return {"samples": samples}, loss
