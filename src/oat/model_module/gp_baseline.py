from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    normal = dist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    scaled = (obs - mu) / sigma

    Phi = normal.cdf(scaled)
    phi = torch.exp(normal.log_prob(scaled))
    crps = sigma * (
        scaled * (2 * Phi - 1)
        + 2 * phi
        - (1 / torch.sqrt(torch.tensor(torch.pi, device=sigma.device)))
    )

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
    mean_groups: List[str] = field(default_factory=lambda: ["nwp", "topo"])
    stationary_cov_groups: List[str] = field(
        default_factory=lambda: [
            "geo",
        ]
    )
    non_stationary_cov_groups: List[str] = field(
        default_factory=lambda: ["geo", "nwp", "topo"]
    )

    d_model: int = 64
    mean_n_hidden_layers: int = 1
    kernel_n_hidden_layers: int = 2
    noise_variance: float = 1.0
    loss_fn: str = "crps"
    num_samples: int = 100
    eps: float = 1e-3


class GPBaseline(nn.Module):
    requires_observation: bool = True

    # Registered as buffers in __init__; annotated here so type checkers know
    # their type through set_output_normalization / _crps_loss.
    output_loc: torch.Tensor
    output_scale: torch.Tensor
    output_is_log: torch.Tensor

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

        mean_indices: List[int] = []
        stationary_covar_indices: List[int] = []
        non_stationary_covar_indices: List[int] = []
        offset = 0

        # geo
        geo_indices = list(range(offset, offset + 3))
        offset += 3
        if "geo" in config.mean_groups:
            mean_indices += geo_indices
        if "geo" in config.stationary_cov_groups:
            stationary_covar_indices += geo_indices
        if "geo" in config.non_stationary_cov_groups:
            non_stationary_covar_indices += geo_indices

        if config.use_nwp:
            n = len(config.nwp_vars)
            nwp_indices = list(range(offset, offset + n))
            offset += n
            if "nwp" in config.mean_groups:
                mean_indices += nwp_indices
            if "nwp" in config.stationary_cov_groups:
                stationary_covar_indices += nwp_indices
            if "nwp" in config.non_stationary_cov_groups:
                non_stationary_covar_indices += nwp_indices
            self.nwp_interpolator: Optional[NWPInterpolator] = NWPInterpolator(
                NWPInterpolatorConfig(field_extent=config.field_extent)
            )
        else:
            self.nwp_interpolator = None

        if config.use_time_emb:
            time_emb_indices = list(range(offset, offset + 4))
            offset += 4
            if "time_emb" in config.mean_groups:
                mean_indices += time_emb_indices
            if "time_emb" in config.stationary_cov_groups:
                stationary_covar_indices += time_emb_indices
            if "time_emb" in config.non_stationary_cov_groups:
                non_stationary_covar_indices += time_emb_indices

        if config.use_topography:
            if config.topography_encoder_kwargs is None:
                raise ValueError(
                    "topography_encoder_kwargs must be provided when "
                    "use_topography=True."
                )
            self.topography_encoder: Optional[CNNTopographyEncoder] = (
                CNNTopographyEncoder(**config.topography_encoder_kwargs)
            )
            n = int(config.topography_encoder_kwargs["out_channels"])
            topo_indices = list(range(offset, offset + n))
            offset += n
            if "topo" in config.mean_groups:
                mean_indices += topo_indices
            if "topo" in config.stationary_cov_groups:
                stationary_covar_indices += topo_indices
            if "topo" in config.non_stationary_cov_groups:
                non_stationary_covar_indices += topo_indices
        else:
            self.topography_encoder = None

        self.feature_dim = offset
        self.mean_indices = tuple(mean_indices)
        self.stationary_covar_indices = tuple(stationary_covar_indices)
        self.non_stationary_covar_indices = tuple(non_stationary_covar_indices)

        # Mean function
        mean_net = MLP(
            dimensions=[len(self.mean_indices), *config.mean_hidden_dims, 1],
            active_dims=mean_indices,
        )
        mean_function = NeuralMean(net=mean_net)

        # Stationary kernel on specified groups. ard_num_dims gives each active
        # dimension its own learned lengthscale (ARD) instead of a single shared
        # one, so e.g. easting/northing/elevation can decorrelate at different rates.
        rbf_kernel = ScaledRBFKernel(
            eps=config.eps,
            active_dims=stationary_covar_indices,
            ard_num_dims=len(stationary_covar_indices) or None,
        )

        # Non-stationary kernel on specified groups (optional)
        if non_stationary_covar_indices:
            kernel_net = MLP(
                dimensions=[
                    len(self.non_stationary_covar_indices),
                    *config.kernel_hidden_dims,
                    config.kernel_embed_dim,
                ],
                active_dims=non_stationary_covar_indices,
            )
            non_stationary_kernel = NeuralKernel(
                net=kernel_net, kernel=ScaledRBFKernel(eps=config.eps)
            )
            kernel_function = rbf_kernel * non_stationary_kernel
        else:
            kernel_function = rbf_kernel

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        with torch.no_grad():
            likelihood.noise = torch.tensor(float(config.noise_variance))

        dummy_x = torch.zeros(2, self.feature_dim)
        dummy_y = torch.zeros(2)

        self.gp = GPModel(
            train_x=dummy_x,
            train_y=dummy_y,
            likelihood=likelihood,
            mean_module=mean_function,
            covar_module=kernel_function,
        )
        self._mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp
        )

        # Output normalization for the data-space CRPS loss. The GP fits a
        # Gaussian in standardized(-log) space; these buffers map its posterior
        # back to raw-log space so the CRPS loss can be computed in data space
        # (see _crps_loss). Set by the predictor via set_output_normalization
        # once the fitted transform exists; saved/restored with the checkpoint.
        self.register_buffer("output_loc", torch.zeros(1))
        self.register_buffer("output_scale", torch.ones(1))
        self.register_buffer("output_is_log", torch.tensor(False))

    def set_output_normalization(
        self, loc: torch.Tensor, scale: torch.Tensor, is_log: bool
    ) -> None:
        """Register the affine map from standardized space back to raw-log space
        for the (single) obs variable: ``log(y + offset) = mu * scale + loc``.

        When ``is_log`` is False the target is not log-transformed (e.g.
        temperature) and the CRPS loss stays a plain Gaussian CRPS in
        standardized space, which is equivalent to the data-space metric under
        an affine transform (CRPS is scale-equivariant).
        """
        self.output_loc = loc.reshape(1).to(self.output_loc.device)
        self.output_scale = scale.reshape(1).to(self.output_scale.device)
        self.output_is_log = torch.tensor(
            bool(is_log), device=self.output_is_log.device
        )

    def _crps_loss(
        self, target: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Mean CRPS between the GP posterior predictive ``N(mu, sigma)`` and
        ``target``, computed in *data space*.

        For a log-transformed target the data-space predictive is a LogNormal;
        its closed-form CRPS is
        ``y·(2Φ(z)−1) − 2m·(Φ(z−σ) − Φ(−σ/√2))`` with ``z = (log y − μ)/σ`` and
        ``m = exp(μ + σ²/2)``. Computing the loss here keeps it aligned with the
        reported data-space CRPS; minimizing the log-space Gaussian CRPS instead
        diverges from the data-space objective (the log-normal tail blows up as
        log-space variance grows).

        Numerical care: the ``m·Φ(·)`` products are evaluated as
        ``exp(log m + log Φ)`` (``log_ndtr``) so a large ``sigma_log`` stays
        finite/differentiable — computing ``m`` directly overflows fp32 and then
        ``m·Φ → ∞·0 = NaN`` (surfaces as NaN loss/grads early in training).
        ``sigma_log`` is floored to avoid a divide-by-zero if the scale
        collapses. ``(y + offset) = exp(mu·scale + loc)`` is strictly positive
        (no ``log(0)``) and, because CRPS is shift-invariant, the shared
        ``+offset`` cancels against the metric.
        """
        if not bool(self.output_is_log):
            return crps_normal(target, mu, sigma)

        loc = self.output_loc.to(mu.device)
        scale = self.output_scale.to(mu.device)
        eps = torch.finfo(mu.dtype).eps
        mu_log = mu * scale + loc
        sigma_log = (sigma * scale).clamp(min=eps)
        y_log = target * scale + loc  # = log(precip + offset)
        y = torch.exp(y_log)  # data-space target (precip + offset) > 0
        z = (y_log - mu_log) / sigma_log
        log_m = mu_log + 0.5 * sigma_log**2
        std_normal = dist.Normal(torch.zeros_like(mu_log), torch.ones_like(sigma_log))
        m_cdf_z = torch.exp(log_m + torch.special.log_ndtr(z - sigma_log))
        m_cdf_ref = torch.exp(log_m + torch.special.log_ndtr(-sigma_log / math.sqrt(2)))
        crps = y * (2 * std_normal.cdf(z) - 1) - 2 * (m_cdf_z - m_cdf_ref)
        return crps.mean()

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
                    total_loss = total_loss + self._crps_loss(
                        ty[valid_tgt], mu[valid_tgt], sigma[valid_tgt]
                    )
                    n_terms += 1

                with torch.no_grad():
                    sample_values = lik_pred.rsample(torch.Size([num_samples]))

            var_samples.append(sample_values.T)

        samples = torch.stack(var_samples, dim=0).unsqueeze(-1)
        loss = total_loss / max(n_terms, 1)
        return {"samples": samples}, loss
