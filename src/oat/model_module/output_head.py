import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod, ABC
from typing import Dict, Any, Tuple

class BaseOutputHead(nn.Module, ABC):
    
    """
    Base class for output heads.
    """
    def __init__(self, *, 
                 in_features: int, 
                 target_features: int, 
                 **kwargs: Any) -> None:
        """
        Args:
            in_features: Channels of input features.
            target_features: Channels of output features.
        """
        super().__init__()
        self.in_features = in_features
        self.target_features = target_features
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the predictions given input features.
        
        Args:
            x: input features [..., in_features]
        Returns:
            Predictions [..., target_features] or dict of tensors
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the innovation (residual) between predictions and target.
        
        Args:
            x: input features [..., in_features]
            target: target values [..., target_features]
            mask: boolean mask indicating valid entries [..., target_features]
            
        Returns:
            Innovation (residuals) [..., target_features]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute the loss between predictions and target, applying the mask.
        
        Args:
            x: input features [..., in_features]
            target: target values [..., target_features]
            mask: boolean mask indicating valid entries [..., target_features]
            
        Returns:
            Predictions dictionary and loss value (scalar tensor)
        """
        raise NotImplementedError()
    
    @abstractmethod
    def sample(self, out: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        """
        Generate samples from the output distribution.
        
        Args:
            out: output dictionary from forward pass
            num_samples: number of samples to generate
        Returns:
            Samples [..., num_samples, target_features]
        """        
        raise NotImplementedError("Sampling not implemented for this output head.")
    
    def loss_normalization_per_sample(self, loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute normalization factor for loss per sample based on mask.
        
        Args:
            loss: loss tensor [B, S, target_features]
            mask: boolean mask indicating valid entries [B, S, target_features]
            
        Returns:
            Normalization factor per sample [B]
        """
        loss_per_sample = torch.sum(loss * mask, dim=[-2, -1])  # [B]
        valid_counts = torch.sum(mask, dim=[-2, -1])  # [B]
        normalization = torch.clamp(valid_counts, min=1.0)  # Avoid division by zero
        norm_loss = torch.mean(loss_per_sample / normalization)  # [B]
        return norm_loss

class LinearOutputHead(BaseOutputHead):
    """
    Linear output head.
    """
    def __init__(self, 
                 *,
                 in_features: int, 
                 target_features: int,
                 loss_fn: str | None = None) -> None:
        """
        Args:
            in_features: Channels of input features.
            target_features: Channels of output features.
            loss_fn: Loss function to use [mse, mae, huber]. Defaults to None (MSE loss).

        Raises:
            ValueError: If an unknown loss function is specified.
        """
        super().__init__(in_features=in_features, target_features=target_features)
        self.net = nn.Linear(self.in_features, 
                             self.target_features)
        if loss_fn is None or loss_fn == "mse":
            self.loss_fn = F.mse_loss
        elif loss_fn == "mae":
            self.loss_fn = F.l1_loss
        elif loss_fn == "huber":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss fn: {loss_fn}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Generate predictions.

        Args:
            x: input [B, S, in_features]

        Returns:
            Predictions {"val_pred": [B, S, target_features]}
        """
        output = self.net(x)
        return {"val_pred": output}  # [B, S, target_features]
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute innovation (residuals).

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]
            mask: boolean mask [B, S, target_features]

        Returns:
            Innovation (residuals) [B, S, target_features]
        """
        out = self.forward(x)
        pred = out['val_pred']
        innovation = target - pred  # [B, S, target_features]
        return mask * innovation
    
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """ Compute loss.

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]
            mask: boolean mask [B, S, target_features]

        Returns:
            Loss value (scalar tensor)
        """
        out = self.forward(x)
        pred = out['val_pred']
        loss = self.loss_fn(pred, target, reduction='none')  # [num_valid]
        loss = self.loss_normalization_per_sample(loss, mask)
        return out, loss
    
    def sample(self, out: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        """ Generate samples from the output distribution.

        Args:
            out: output dictionary from forward pass
            num_samples: number of samples to generate
        Returns:
            Samples [B, S, num_samples, target_features]
        """
        pred = out['val_pred']  # [B, S, target_features]
        samples = pred.unsqueeze(-2)  # [B, S, num_samples, target_features]
        return samples
    
class GaussianOutputHead(BaseOutputHead):
    
    def __init__(self, 
                *,
                in_features: int, 
                target_features: int,
                std_fn: str | None = "softplus",
                min_std_val: float | None = 1e-3
                ) -> None:
        """
        Args:
            in_features: Channels of input features.
            target_features: Channels of output features.
            std_fn: Function used to transform the raw standard deviation (e.g. "softplus", "exp"). Defaults to "softplus".
            min_std_val: Minimum standard deviation value used to clamp the output. If None, no clamping is applied. Defaults to 0.

        Raises:
            ValueError: If an unknown std_fn is specified.
        """
        super().__init__(in_features=in_features, target_features=target_features)
        self.net = nn.Linear(self.in_features, 
                             2 * self.target_features)
        if std_fn == "softplus":
            self.std_fn = nn.Softplus()
        elif std_fn == "exp":
            self.std_fn = torch.exp
        else:
            raise ValueError(f"Unknown std fn: {std_fn}")
        self.min_std_val = min_std_val
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Generate predictions.

        Args:
            x: input [..., in_features]

        Returns:
            Predictions dict with 'mean' and 'std' tensors, each of shape [B, S, target_features]
        """
        output = self.net(x)
        mean, raw_std = torch.chunk(output, 2, dim=-1)
        std = self.std_fn(raw_std)
        if self.min_std_val is not None:
            std = torch.clamp(std, min=self.min_std_val)
        return {"mu_pred": mean, "std_pred": std}  # [B, S, target_features]
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute innovation (residuals).

        Args:
            x: input [..., in_features]
            target: target values [..., target_features]
            mask: boolean mask [..., target_features]

        Returns:
            Innovation (residuals) [..., target_features]
        """
        preds = self.forward(x)
        mean = preds["mu_pred"]
        std = preds["std_pred"]
        innovation = (target - mean) / std  # [..., target_features]
        return mask * innovation
    
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """ Compute negative log-likelihood loss.

        Args:
            x: input [..., in_features]
            target: target values [..., target_features]
            mask: boolean mask [..., target_features]
        Returns:
            Loss value (scalar tensor)
        """
        preds = self.forward(x)
        mean = preds["mu_pred"]
        std = preds["std_pred"]
        var = std ** 2
        nll = torch.log(std) + 0.5 * ((target - mean) ** 2) / var
        #loss = torch.sum(nll[mask]) / torch.minimum(torch.sum(mask), torch.tensor(1.0))
        loss = self.loss_normalization_per_sample(nll, mask)
        return preds, loss
    
    
    def sample(self, out: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
        """ Generate samples from the output distribution.

        Args:
            out: output dictionary from forward pass
            num_samples: number of samples to generate
        Returns:
            Samples [..., num_samples, target_features]
        """
        # TODO: Check that this is really sampling from the distribution
        mean = out['mu_pred']  # [..., target_features]
        std = out['std_pred']  # [..., target_features]
        leading_shape = mean.shape[:-1]
        F = mean.shape[-1]
        
        # Expand mean and std for sampling
        mean = mean.unsqueeze(-2).expand(*leading_shape, num_samples, F)  # [..., num_samples, target_features]
        std = std.unsqueeze(-2).expand(*leading_shape, num_samples, F)    # [..., num_samples, target_features]
        
        # Sample from Gaussian
        eps = torch.randn_like(std)  # [..., num_samples, target_features]
        samples = mean + eps * std   # [..., num_samples, target_features]
        
        return samples
    
    
class QuantileOutputHead(BaseOutputHead):
    """
    Quantile output head using Bernstein polynomial parameterization.
    Produces monotone quantile functions via cumulative sum of positive coefficients.
    """

    def __init__(
        self,
        *,
        in_features: int,
        target_features: int,
        degree: int = 20,
        non_linearity: nn.Module | None = None,
        min_val: float | None = None,
        train_taus: torch.Tensor | None = None,
    ):
        """
        Args:
            in_features: Input feature dimensions
            target_features: Target dimension
            degree: Number of Bernstein basis functions (default: 20)
            non_linearity: Activation for positive coefficients (default: Softplus)
            min_val: Fixed minimum quantile value (optional)
            train_taus: Training quantile levels (default: linspace(0, 1, 10))
        """
        super().__init__(in_features=in_features, target_features=target_features)
        self.degree = degree

        # Coefficient network with non-linearity for positivity
        self.non_linearity = non_linearity or nn.Softplus()
        self.coeff_net = nn.Sequential(
            nn.Linear(in_features, (degree + 1) * target_features),
            self.non_linearity,
        )

        # Minimum value: either fixed or learned
        if min_val is not None:
            self.register_buffer("min_val", torch.tensor(float(min_val)))
            self.min_val_net = None
        else:
            self.min_val_net = nn.Linear(in_features, target_features)
            self.register_buffer("min_val", None)

        # Default training quantile grid
        if train_taus is None:
            train_taus = torch.linspace(0.0, 1.0, 10)
        self.register_buffer("taus", train_taus)
        
        # Precompute quantile spacings for integration
        dtau = torch.diff(
            torch.cat([torch.zeros(1, device=train_taus.device), train_taus])
        )
        self.register_buffer("dtau", dtau)

        # Precompute binomial coefficients: C(n, i)
        binom = torch.tensor(
            [math.comb(degree, i) for i in range(degree + 1)], 
            dtype=torch.float32
        )
        self.register_buffer("binom", binom)

    def _bernstein_basis(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute Bernstein basis polynomials:
            B_{i,n}(tau) = C(n,i) * tau^i * (1-tau)^(n-i)
        
        Args:
            tau: Quantile levels [..., Q]
        
        Returns:
            Basis matrix [..., Q, degree+1]
        """
        tau = tau.unsqueeze(-1)  # [..., Q, 1]
        i = torch.arange(
            self.degree + 1, 
            device=tau.device, 
            dtype=tau.dtype
        )  # [degree+1]
        
        # B_{i,n}(tau) = C(n,i) * tau^i * (1-tau)^(n-i)
        binom = self.binom.to(device=tau.device, dtype=tau.dtype)
        basis = binom * (tau ** i) * ((1 - tau) ** (self.degree - i))
        
        return basis  # [..., Q, degree+1]

    def forward(
        self, 
        x: torch.Tensor, 
        tau: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantile values for given inputs and quantile levels.

        Args:
            x: Input features [..., in_features]
            tau: Quantile levels [..., Q] (if None, uses self.taus)

        Returns:
            Quantile predictions [..., Q, target_features]
        """
        leading_shape = x.shape[:-1]
        
        # Use provided tau or default training taus
        if tau is None:
            tau = self.taus.to(device=x.device, dtype=x.dtype)
            tau = tau.view(*([1] * len(leading_shape)), -1)
            tau = tau.expand(*leading_shape, -1)
        
        # Validate tau range
        tau_min = tau.min().item()
        tau_max = tau.max().item()
        if tau_min < 0.0 or tau_max > 1.0:
            raise ValueError("tau must be in [0, 1]")

        # Compute Bernstein basis
        Bmat = self._bernstein_basis(tau)  # [..., Q, degree+1]
        # Get positive incremental coefficients and cumsum for monotonicity
        raw_coeffs = self.coeff_net(x)  # [..., (degree+1)*target_features]
        raw_coeffs = raw_coeffs.view(
            *leading_shape, 
            self.degree + 1, 
            self.target_features
        )  # [..., degree+1, target_features]
        coeffs = torch.cumsum(raw_coeffs, dim=-2)  # Monotonicity constraint
        
        # Compute minimum value
        if self.min_val is not None:
            # Fixed minimum
            min_vals = self.min_val.view(
                *([1] * (len(leading_shape) + 1)), 
                self.target_features
            )
            min_vals = min_vals.expand(*leading_shape, 1, self.target_features)
        else:
            # Learned minimum
            min_vals = self.min_val_net(x)  # [..., target_features]
            min_vals = min_vals.unsqueeze(-2)  # [..., 1, target_features]

        # Quantile function: Q(tau) = min_val + sum_i(c_i * B_i(tau))
        quantiles = min_vals + torch.einsum(
            "...qk,...kn->...qn", 
            Bmat, 
            coeffs
        )  # [..., Q, target_features]
        
        return {
            "qval_pred": quantiles,
            "tau": tau,
            "coeffs": coeffs,
            "min_vals": min_vals,
        }  # [..., Q, target_features]
    
    def _compute_inv_cdf(
        self,
        q: torch.Tensor,
        coeffs: torch.Tensor,
        min_vals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the inverse CDF (quantile function) for given quantiles and coefficients.
        
        Args:
            q: Quantile levels [..., Q]
            coeffs: Coefficients of Bernstein basis [..., degree+1, target_features]
            min_vals: Optional minimum value offset [..., target_features] to be added
                to the Bernstein polynomial sum, i.e.
                    Q(tau) = min_vals + sum_i c_i * B_i(tau)
        
        Returns:
            Quantile values [..., Q, target_features]
        """
        
        basis = self._bernstein_basis(q)  # [..., Q, degree+1]
        quantiles = torch.einsum(
            "...qk,...kn->...qn",
            basis,
            coeffs,
        )  # [..., Q, target_features]

        if min_vals is not None:
            # Broadcast min_vals [..., target_features] across the quantile dimension.
            quantiles = quantiles + min_vals.unsqueeze(-2)
        return quantiles
    
    def compute_innovation(
        self, 
        x: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standardized innovation per equation (11):
            δy = (y - Q_50) / (Q_75 - Q_25)
        
        Args:
            x: Input features [..., in_features]
            target: Observed values [..., target_features]
            mask: Observation mask [..., target_features]
        
        Returns:
            Masked innovation [..., target_features]
        """
        leading_shape = x.shape[:-1]
        
        # Quantile levels: 25%, 50%, 75%
        taus = torch.tensor(
            [0.25, 0.5, 0.75], 
            device=x.device, 
            dtype=x.dtype
        )
        tau = taus.view(*([1] * len(leading_shape)), -1)
        tau = tau.expand(*leading_shape, -1)
        
        # Get quantile predictions
        out = self.forward(x, tau)  # [..., 3, target_features]
        preds = out["qval_pred"]  # [..., 3, target_features]
        q25 = preds[..., 0, :]   # [..., target_features]
        q50 = preds[..., 1, :]   # [..., target_features]
        q75 = preds[..., 2, :]   # [..., target_features]
        
        # Standardized innovation: (y - median) / IQR
        iqr = q75 - q25
        innovation = (target - q50) / (iqr + 1e-6)
        
        return mask * innovation

    def loss(
        self, 
        x: torch.Tensor, 
        target: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Integrated quantile regression loss using trapezoidal integration.
        
        Approximates: ∫_0^1 ρ_τ(Y - Q_τ(X)) dτ
        
        where ρ_τ(u) = u(τ - 1{u < 0}) is the check function (pinball loss).
        
        Args:
            x: Input features [..., in_features]
            target: Observed values [..., target_features]
            mask: Observation mask [..., target_features]
            
        Returns:
            Scalar loss value
        """
        leading_shape = x.shape[:-1]
        
        # Get training quantiles
        taus = self.taus.to(device=x.device, dtype=x.dtype)
        tau = taus.view(*([1] * len(leading_shape)), -1).expand(*leading_shape, -1)
        
        # Predict quantiles
        out = self.forward(x, tau)  # [..., Q, target_features]
        preds = out["qval_pred"]
        # Expand target and mask
        target = target.unsqueeze(-2)  # [..., 1, target_features]
        mask = mask.unsqueeze(-2)      # [..., 1, target_features]
        
        # Compute errors
        errors = preds - target  # [..., Q, target_features]
        
        # Check function: ρ_τ(e) = e * (τ - 1{e < 0})
        tau_expanded = tau.unsqueeze(-1)  # [..., Q, 1]
        check_loss = errors * (tau_expanded - (errors < 0).float())
        
        # Apply mask
        check_loss = check_loss * mask  # [..., Q, target_features]
        
        # Integrate over quantile dimension using trapezoidal rule
        # torch.trapezoid integrates along the specified dimension
        integrated_loss = torch.trapezoid(check_loss, taus, dim=-2)  # [..., target_features]
        
        # Normalize by number of valid observations
        #total_loss = integrated_loss.sum() / mask.sum().clamp(min=1.0)
        loss = self.loss_normalization_per_sample(integrated_loss, mask.squeeze(-2))
        return out, loss
    
    
    def sample(
        self, 
        out: Dict, 
        num_samples: int
    ) -> torch.Tensor:
        """
        Generate samples from the quantile function.

        Args:
            out: output dictionary from forward pass
            num_samples: number of samples to generate
        Returns:
            Samples [..., num_samples, target_features]
        """
        
        # TODO: Check that this is really sampling from the distribution
        coeffs = out["coeffs"]  # [..., degree+1, target_features]
        leading_shape = coeffs.shape[:-2]
        
        # Sample uniform quantile levels
        q = torch.rand(
            *leading_shape, 
            num_samples, 
            device=coeffs.device, 
            dtype=coeffs.dtype
        )  # [..., num_samples]
        
        # Compute quantile values for sampled quantiles
        samples = self._compute_inv_cdf(q, coeffs)  # [..., num_samples, target_features]
        
        return samples