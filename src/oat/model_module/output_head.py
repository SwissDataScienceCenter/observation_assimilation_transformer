import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import abstractmethod, ABC

class BaseOutputHead(nn.Module, ABC):
    
    """
    Base class for decoders.
    """
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Get the predictions given input features.
        
        Args:
            x: input features [B, S, in_features]
        Returns:
            Predictions [B, S, target_features]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the innovation (residual) between predictions and target.
        
        Args:
            x: input features [B, S, in_features]
            target: target values [B, S, target_features]
            
        Returns:
            Innovation (residuals) [B, S, target_features]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between predictions and target, applying the mask.
        
        Args:
            x: input features [B, S, in_features]
            target: target values [B, S, target_features]
            mask: boolean mask indicating valid entries [B, S, target_features]
            
        Returns:
            Loss value (scalar tensor)
        """
        raise NotImplementedError()

class LinearOutputHead(BaseOutputHead):
    """
    Linear output head.
    """
    def __init__(self, 
                 in_features: int, 
                 target_features: int,
                 loss_fn: str | None = None) -> None:
        """
        Args:
            in_features: Channels of input features.
            out_features: Channels of output features.
            loss_fn: Loss function to use [mse, mae, huber]. Defaults to None (MSE loss).

        Raises:
            ValueError: If an unknown loss function is specified.
        """
        super().__init__()
        self.in_features = in_features
        self.target_features = target_features
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Generate predictions.

        Args:
            x: input [B, S, in_features]

        Returns:
            Predictions [B, S, target_features]
        """
        output = self.net(x)
        assert isinstance(output, torch.Tensor)
        return output  # [B, S, target_features]
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute innovation (residuals).

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]

        Returns:
            Innovation (residuals) [B, S, target_features]
        """
        pred = self.forward(x)
        innovation = target - pred  # [B, S, target_features]
        return mask * innovation
    
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute loss.

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]
            mask: boolean mask [B, S, target_features]

        Returns:
            Loss value (scalar tensor)
        """
        pred = self.forward(x)
        loss = self.loss_fn(pred[mask], target[mask])
        return loss
    
class GaussianOutputHead(BaseOutputHead):
    
    def __init__(self, 
                in_features: int, 
                target_features: int,
                std_fn: str | None = "softplus",
                min_val: float | None = 0
                ) -> None:
        """
        Args:
            in_features: Channels of input features.
            target_features: Channels of output features.
            loss_fn: Loss function to use [mse, mae, huber]. Defaults to None (MSE loss).

        Raises:
            ValueError: If an unknown loss function is specified.
        """
        super().__init__()
        self.in_features = in_features
        self.target_features = target_features
        self.net = nn.Linear(self.in_features, 
                             2 * self.target_features)
        if std_fn == "softplus":
            self.std_fn = nn.Softplus()
        elif std_fn == "exp":
            self.std_fn = torch.exp
        else:
            raise ValueError(f"Unknown std fn: {std_fn}")
        self.min_val = min_val
        
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """ Generate predictions.

        Args:
            x: input [B, S, in_features]

        Returns:
            Predictions dict with 'mean' and 'std' tensors, each of shape [B, S, target_features]
        """
        output = self.net(x)
        mean, raw_std = torch.chunk(output, 2, dim=-1)
        std = self.std_fn(raw_std)
        if self.min_val is not None:
            std = torch.clamp(std, min=self.min_val)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)
        return {"mean": mean, "std": std}  # [B, S, target_features]
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute innovation (residuals).

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]

        Returns:
            Innovation (residuals) [B, S, target_features]
        """
        preds = self.forward(x)
        mean = preds["mean"]
        std = preds["std"]
        innovation = (target - mean) / std  # [B, S, target_features]
        return mask * innovation
    
    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Compute negative log-likelihood loss.

        Args:
            x: input [B, S, in_features]
            target: target values [B, S, target_features]
            mask: boolean mask [B, S, target_features]

        Returns:
            Loss value (scalar tensor)
        """
        preds = self.forward(x)
        mean = preds["mean"]
        std = preds["std"]
        var = std ** 2
        nll = torch.log(std) + 0.5 * ((target - mean) ** 2) / var
        loss = torch.sum(nll[mask]) / torch.minimum(torch.sum(mask), torch.tensor(1.0))
        return loss
    
    
class QuantileOutputHead(BaseOutputHead):
    """
    Quantile decoder using Bernstein polynomial parameterization.
    Produces monotone quantile functions.
    """

    def __init__(
        self,
        in_features: int,
        target_features: int,
        degree: int,
        non_linearity: nn.Module | None = None,
        min_val: float | None = None,
        train_taus: torch.Tensor | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.target_features = target_features
        self.degree = degree

        # Nonlinearity for coefficient network
        self.non_linearity = non_linearity or nn.Softplus()

        # Coeff network (degree+1 Bernstein coefficients)
        self.coeff_net = nn.Sequential(
            nn.Linear(in_features, (degree + 1) * target_features),
            self.non_linearity,
        )

        # Minimum quantile value handling
        if min_val is not None:
            self.register_buffer("min_val", torch.tensor(float(min_val)))
            self.min_val_net = None
        else:
            self.min_val_net = nn.Linear(in_features, target_features)
            self.min_val = None

        # Quantile grid
        if train_taus is None:
            train_taus = torch.linspace(0.0, 1.0, 10)
        self.register_buffer("taus", train_taus)
        dtau = torch.diff(torch.cat([torch.zeros(1, device=train_taus.device), train_taus]))
        self.register_buffer("dtau", dtau)

        # Precompute binomial coefficients
        combs = torch.tensor(
            [math.comb(degree, i) for i in range(degree + 1)], dtype=torch.float32
        )
        self.register_buffer("binom", combs)

    def _bernstein_basis(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Bernstein basis:
            B_{i,n}(tau) = C(n,i) * tau^i * (1 - tau)^(n-i)
        tau: [B, Q]
        Returns: [B, Q, degree+1]
        """
        tau = tau.unsqueeze(-1)                   # [B, Q, 1]
        # use the same dtype as tau for proper broadcasting with powers
        i = torch.arange(self.degree + 1, device=tau.device, dtype=tau.dtype)  # [degree+1]

        # ensure binom is a tensor on the same device and dtype (handles static/type-checker issues)
        binom = self.binom
        if not isinstance(binom, torch.Tensor):
            binom = torch.tensor(binom, device=tau.device, dtype=tau.dtype)
        else:
            binom = binom.to(device=tau.device, dtype=tau.dtype)

        basis = binom * (tau ** i) * ((1 - tau) ** (self.degree - i))
        return basis  # [..., Q, degree+1]

    def forward(self, x: torch.Tensor, tau: torch.Tensor | None = None) -> torch.Tensor:
        
        leading_shape = x.shape[:-1]
        
        if not isinstance(self.taus, torch.Tensor):
            taus = torch.tensor(self.taus, device=x.device)
        else:
            taus = self.taus.to(device=x.device)
        
        tau = taus.view(*([1] * len(leading_shape)), -1).expand(*leading_shape, -1)
        
        assert 0.0 <= tau.min() and tau.max() <= 1.0, "taus must be in [0, 1]"

        # Bernstein basis
        Bmat = self._bernstein_basis(tau)  # [..., Q, degree+1]

        # Positive increments via softplus + cumsum ensures monotonicity
        raw_coeffs = self.coeff_net(x)                          # [..., (degree+1)*target_features]
        raw_coeffs = raw_coeffs.view(*leading_shape, self.degree + 1, self.target_features)  # [..., degree+1, target_features]
        coeffs = torch.cumsum(raw_coeffs, dim=-2)
        
        # Min value
        if self.min_val is not None:
            min_vals = self.min_val.view(*([1] * (len(leading_shape) + 1)), self.target_features)
            min_vals = min_vals.expand(*leading_shape, 1, self.target_features)
        else:
            assert self.min_val_net is not None
            min_vals = self.min_val_net(x)                      # [..., target_features]
            min_vals = min_vals.unsqueeze(-2)                    # [..., 1, target_features]

        out = min_vals + torch.einsum("...qk,...kn->...qn", Bmat, coeffs)  # [..., Q, target_features]
        assert isinstance(out, torch.Tensor)
        return out  # [..., Q, target_features]
    
    def compute_innovation(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Integrated quantile regression innovation.
        """
        leading_shape = x.shape[:-1]
        # tau is [.25, .5, .75]
        taus = torch.tensor([0.25, 0.5, 0.75], device=x.device, dtype=x.dtype)
        tau = taus.view(*([1] * len(leading_shape)), -1).expand(*leading_shape, -1)
        
        preds = self.forward(x, tau)
        pred25 = preds[..., 0, :]  # [..., 1, target_features]
        mean_pred = preds[..., 1, :]  # [..., 1, target_features]
        pred75 = preds[..., 2, :]  # [..., 1, target_features]
        
        innovation = (target - mean_pred) / (pred75 - pred25 + 1e-6)  # [B, S, target_features]
        return mask * innovation

    def loss(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Integrated quantile regression loss.
        """
        leading_shape = x.shape[:-1]
        
        if not isinstance(self.taus, torch.Tensor):
            taus = torch.tensor(self.taus, device=x.device, dtype=x.dtype)
        else:
            taus = self.taus.to(device=x.device, dtype=x.dtype)
        
        if not isinstance(self.dtau, torch.Tensor):
            dtau = torch.tensor(self.dtau, device=x.device, dtype=x.dtype)
        else:
            dtau = self.dtau.to(device=x.device, dtype=x.dtype)
        
        # Expand taus and dtau to match leading dimensions
        tau = taus.view(*([1] * len(leading_shape)), -1).expand(*leading_shape, -1)
        dtau_expanded = dtau.view(*([1] * len(leading_shape)), -1).expand(*leading_shape, -1)
        
        preds = self.forward(x, tau)  # [..., Q, target_features]
        
        target = target.unsqueeze(-2)  # [..., 1, target_features]
        mask = mask.unsqueeze(-2)  # [..., 1, target_features]
        
        errors = preds - target  # [..., Q, target_features]
        weight = (errors >= 0).float() - tau.unsqueeze(-1)  # [..., Q, target_features]
        
        # Apply mask and compute loss
        dtau_expanded = dtau_expanded.unsqueeze(-1)  # [..., Q, 1]
        loss_per_sample = weight * errors * dtau_expanded  # [..., Q, target_features]
        loss_per_sample = loss_per_sample * mask  # Apply mask
        
        loss = 2 * loss_per_sample.sum() / mask.sum().clamp(min=1.0)
        return loss
