from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import gpytorch
import torch
import torch.nn as nn
from linear_operator import settings as linear_operator_settings
from linear_operator.utils.errors import NotPSDError
from linear_operator.utils.warnings import NumericalWarning
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

# Default cholesky_max_tries (3) escalates jitter only up to 1e-4 before
# raising NotPSDError; we've observed context covariances with min eigenvalue
# beyond that (e.g. -0.087), so give the solver more headroom to recover
# before giving up. Applied around the whole per-batch-element loop rather
# than per-call so every Cholesky solve in that scope gets the same budget.
_CHOLESKY_MAX_TRIES = 6


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
    debug_diagnostics: bool = False
    noise_lower_bound: float = 1e-2
    outputscale_lower_bound: float = 1e-2
    stationary_lengthscale_lower_bound: float = 1e-2


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
        # Lower bounds on lengthscale/outputscale keep K(cx,cx) from drifting into
        # the severely ill-conditioned regime observed during training (see
        # debug/gp-training-instability): unconstrained, noise/outputscale collapse
        # over epochs until even jittered Cholesky solves fail.
        #
        # Only the stationary kernel's outputscale is trainable. The overall
        # kernel is a product (rbf_kernel * non_stationary_kernel), so only the
        # *product* of the two outputscales is identifiable from data -- having
        # both trainable is a redundant degree of freedom the optimizer can
        # wander along without changing model behavior. The non-stationary
        # kernel's outputscale is fixed at 1 below.
        rbf_kernel = ScaledRBFKernel(
            eps=config.eps,
            active_dims=stationary_covar_indices,
            ard_num_dims=len(stationary_covar_indices) or None,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(
                config.stationary_lengthscale_lower_bound
            ),
            outputscale_constraint=gpytorch.constraints.GreaterThan(
                config.outputscale_lower_bound
            ),
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
            # Lengthscale and outputscale both fixed (not learned): this RBF
            # operates on the MLP's learned embedding, and rescaling that
            # embedding by c is mathematically equivalent to rescaling the
            # lengthscale by 1/c and the square root of the outputscale by
            # c -- the network can freely undo either value via its own
            # output scale, so leaving them trainable is a redundant degree
            # of freedom (see the outputscale note above). Standard
            # deep-kernel-learning practice fixes both.
            non_stationary_kernel = NeuralKernel(
                net=kernel_net,
                kernel=ScaledRBFKernel(
                    eps=config.eps,
                    lengthscale=torch.ones(1),
                    train_lengthscale=False,
                    variance=torch.tensor(1.0),
                    train_variance=False,
                ),
            )
            kernel_function = rbf_kernel * non_stationary_kernel
        else:
            kernel_function = rbf_kernel

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(config.noise_lower_bound)
        )
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
        self._diag: Dict[str, List[float]] = {}

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

    def _record_context_diagnostics(self, cx: torch.Tensor) -> None:
        with torch.no_grad():
            kxx = self.gp.covar_module(cx).to_dense()
            try:
                eigvals = torch.linalg.eigvalsh(kxx)
                min_eig = eigvals.min().item()
                cond_number = eigvals.max().item() / max(min_eig, 1e-12)
                eig_failed = 0
            except RuntimeError:
                # Dense eigh itself failing to converge is a stronger signal
                # than a large condition number: the matrix is degenerate
                # enough that even a robust solver can't handle it. Record a
                # sentinel rather than letting this diagnostic crash training.
                min_eig = 0.0
                cond_number = 1e12
                eig_failed = 1
        self._diag.setdefault("min_eig", []).append(min_eig)
        self._diag.setdefault("cond_number", []).append(cond_number)
        self._diag.setdefault("eig_failure_count", []).append(eig_failed)

    def _posterior_predictive(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        tx: torch.Tensor,
    ) -> gpytorch.distributions.base_distributions.Normal:
        self.gp.set_train_data(inputs=cx, targets=cy, strict=False)
        self.gp.eval()
        self.gp.likelihood.eval()
        if self.config.debug_diagnostics:
            self._record_context_diagnostics(cx)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", NumericalWarning)
                with gpytorch.settings.fast_pred_var():
                    pred = self.gp(tx)
            cg_count = sum("CG terminated" in str(w.message) for w in caught)
            jitter_count = sum("added jitter" in str(w.message) for w in caught)
            self._diag.setdefault("cg_nonconverge_count", []).append(cg_count)
            self._diag.setdefault("jitter_count", []).append(jitter_count)
        else:
            with gpytorch.settings.fast_pred_var():
                pred = self.gp(tx)
        return self.gp.likelihood(pred)

    def _mll_loss_for_context(
        self, cx: torch.Tensor, cy: torch.Tensor
    ) -> Optional[torch.Tensor]:
        self.gp.set_train_data(inputs=cx, targets=cy, strict=False)
        self.gp.train()
        self.gp.likelihood.train()
        try:
            output = self.gp(cx)
            return -self._mll(output, cy)
        except NotPSDError:
            # The context covariance is occasionally too ill-conditioned for
            # the exact Cholesky the marginal likelihood needs. Skip this
            # batch element's contribution rather than losing the whole run.
            return None

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

        with linear_operator_settings.cholesky_max_tries(_CHOLESKY_MAX_TRIES):
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

        if self.config.debug_diagnostics:
            self._diag = {}

        with linear_operator_settings.cholesky_max_tries(_CHOLESKY_MAX_TRIES):
            for batch_idx in range(batch_size):
                valid_ctx = context_mask[batch_idx].bool()
                cx = context_x[batch_idx][valid_ctx]
                cy = context_y[batch_idx][valid_ctx]
                tx = target_x[batch_idx]
                ty = target_y[batch_idx]
                valid_tgt = target_mask[batch_idx].bool()
                cx, cy = self._ensure_non_empty_context(cx, cy, tx)

                if self.config.loss_fn == "mll":
                    mll_loss = self._mll_loss_for_context(cx, cy)
                    if mll_loss is not None:
                        total_loss = total_loss + mll_loss
                        n_terms += 1
                    if self.config.debug_diagnostics:
                        self._diag.setdefault("mll_failure_count", []).append(
                            0 if mll_loss is not None else 1
                        )
                    with torch.no_grad():
                        lik_pred = self._posterior_predictive(cx, cy, tx)
                        if self.config.debug_diagnostics:
                            sigma = lik_pred.stddev
                            self._diag.setdefault("sigma_min", []).append(
                                sigma.min().item()
                            )
                            self._diag.setdefault("sigma_max", []).append(
                                sigma.max().item()
                            )
                        sample_values = lik_pred.rsample(torch.Size([num_samples]))
                else:
                    lik_pred = self._posterior_predictive(cx, cy, tx)
                    mu = lik_pred.mean
                    sigma = lik_pred.stddev

                    if self.config.debug_diagnostics:
                        self._diag.setdefault("sigma_min", []).append(
                            sigma.min().item()
                        )
                        self._diag.setdefault("sigma_max", []).append(
                            sigma.max().item()
                        )

                    if valid_tgt.any():
                        crps = crps_normal(
                            ty[valid_tgt], mu[valid_tgt], sigma[valid_tgt]
                        )
                        total_loss = total_loss + crps.mean()
                        n_terms += 1

                    with torch.no_grad():
                        sample_values = lik_pred.rsample(torch.Size([num_samples]))

                var_samples.append(sample_values.T)

        samples = torch.stack(var_samples, dim=0).unsqueeze(-1)
        loss = total_loss / max(n_terms, 1)
        return {"samples": samples}, loss

    def pop_diagnostics(self) -> Dict[str, float]:
        """Aggregate and clear the per-step diagnostics accumulated across
        the batch loop in `forward_with_loss`/`_posterior_predictive`.

        Worst-case reductions are used deliberately (e.g. min-of-mins for
        eigenvalues, max-of-maxes for condition number) so a single bad
        batch element isn't averaged away.
        """
        diag = self._diag
        self._diag = {}
        reductions: Dict[str, Callable[[Iterable[float]], float]] = {
            "min_eig": min,
            "cond_number": max,
            "sigma_min": min,
            "sigma_max": max,
            "cg_nonconverge_count": sum,
            "jitter_count": sum,
            "eig_failure_count": sum,
            "mll_failure_count": sum,
        }
        return {
            key: float(reductions[key](values))
            for key, values in diag.items()
            if values
        }

    def diagnostic_hyperparams(self) -> Dict[str, float]:
        """Read current kernel/likelihood hyperparameters for diagnostic
        logging. Shared parameters, so this is read once per step (not
        per batch element).
        """
        covar_module = self.gp.covar_module
        stationary_kernel = covar_module
        non_stationary_kernel: Optional[ScaledRBFKernel] = None
        if isinstance(covar_module, gpytorch.kernels.ProductKernel):
            stationary_kernel, deep_kernel = covar_module.kernels
            non_stationary_kernel = deep_kernel.kernel

        out: Dict[str, float] = {
            "noise": self.gp.likelihood.noise.mean().item(),
            "stationary_outputscale": stationary_kernel.kernel.outputscale.mean().item(),
            "stationary_lengthscale_min": stationary_kernel.kernel.base_kernel.lengthscale.min().item(),
            "stationary_lengthscale_max": stationary_kernel.kernel.base_kernel.lengthscale.max().item(),
        }
        if non_stationary_kernel is not None:
            out["neural_outputscale"] = (
                non_stationary_kernel.kernel.outputscale.mean().item()
            )
            out["neural_lengthscale_min"] = (
                non_stationary_kernel.kernel.base_kernel.lengthscale.min().item()
            )
            out["neural_lengthscale_max"] = (
                non_stationary_kernel.kernel.base_kernel.lengthscale.max().item()
            )
        return out
