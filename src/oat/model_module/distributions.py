import math
from collections.abc import Callable

import torch
from torch import distributions as dist
from torch.distributions import transform_to, Distribution, constraints


def check_if_valid_distribution(dist_class: type[Distribution]):
    if not all([constraint.event_dim == 0 for constraint in dist_class.arg_constraints.values()]):
        raise ValueError(f"{dist_class} has multivariate parameters, which are not supported.")
    # OATDistribution subclasses that override crps() analytically don't need rsample().
    has_analytical_crps = (
        issubclass(dist_class, OATDistribution)
        and dist_class.crps is not OATDistribution.crps
    )
    if not dist_class.has_rsample and not has_analytical_crps:
        raise ValueError(
            f"{dist_class} does not support rsample() and provides no analytical crps() override."
        )
    
    # Base torch distributions (e.g. torch.distributions.Normal) typically do not
    # define a custom `name` attribute. We allow those classes here because they
    # are either wrapped by a registered OATDistribution or by a generic wrapper.
    
    # check can calulate innovation, which requires a cdf() method or an override of innovation() itself
    if not hasattr(dist_class, "cdf") and not (issubclass(dist_class, OATDistribution) and dist_class.innovation is not OATDistribution.innovation):
        raise ValueError(
            f"{dist_class} must have a cdf() method or an override of innovation() to compute innovations."
        )


class OATDistribution(Distribution):
    """Base class for OAT-specific distribution extensions.

    Adds a generic MC-CRPS method as default; subclasses may override with
    analytical solutions. Future distribution-level utilities (e.g. PIT,
    energy score) belong here.
    """
    
    name: str | None = None

    def crps(self, target: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        # TODO to double check with oat.predictor_module.functional_metrics.crps_mc
        
        """MC-CRPS:  E|X-y| - 0.5 * E|X-X'|  via differentiable rsample.

        Args:
            target: observed values, shape matching batch_shape
            num_samples: number of rsample draws for MC estimation
        Returns:
            CRPS per element, shape matching batch_shape
        """
        if not self.has_rsample:
            raise RuntimeError(
                f"{type(self).__name__} does not support rsample(); "
                "CRPS requires differentiable sampling."
            )
        samples = self.rsample(sample_shape=(num_samples,))  # [n, *batch]
        first_term = torch.abs(samples - target.unsqueeze(0)).mean(0)
        xi = samples.unsqueeze(0)   # [1, n, *batch]
        xj = samples.unsqueeze(1)   # [n, 1, *batch]
        second_term = torch.abs(xi - xj).sum((0, 1)) / (num_samples * (num_samples - 1))
        return first_term - 0.5 * second_term
    
    def innovation(self, target: torch.Tensor) -> torch.Tensor:
        """Compute the innovation by projecting the target through the distribution's CDF and then applying the inverse CDF of a standard normal distribution."""
        
        # Get the CDF of the distribution at the target value
        cdf_values = self.cdf(target)
        eps = torch.finfo(cdf_values.dtype).eps
        cdf_values = cdf_values.clamp(min=eps, max=1 - eps)
        
        # Apply the inverse CDF (quantile function) of a standard normal distribution to the CDF values
        std_normal = dist.Normal(torch.zeros_like(cdf_values), torch.ones_like(cdf_values))
        innovation_values = std_normal.icdf(cdf_values)
        
        return innovation_values


_REGISTRY: dict[type[Distribution], type[OATDistribution]] = {}
_ALL_DISTRIBUTIONS: set[type[OATDistribution]] = set()

def register_oat_distribution(base: type[Distribution] | None = None):
    def decorator(cls: type[OATDistribution]) -> type[OATDistribution]:
        # always track as available distribution
        _ALL_DISTRIBUTIONS.add(cls)

        # optionally map from base torch distribution
        if base is not None:
            _REGISTRY[base] = cls

        return cls
    return decorator


def as_oat_distribution(dist_class: type[Distribution]) -> type[OATDistribution]:
    """Return an OATDistribution subclass for dist_class.

    Resolution order:
    1. If dist_class is already an OATDistribution subclass, return it.
    2. If dist_class has a registered subclass (e.g. NormalOAT for dist.Normal), return that.
    3. Otherwise generate a generic wrapper via multiple inheritance.
    """
    if issubclass(dist_class, OATDistribution):
        return dist_class  # type: ignore[return-value]
    if dist_class in _REGISTRY:
        return _REGISTRY[dist_class]
    return type(dist_class.__name__, (OATDistribution, dist_class), {})  # type: ignore[return-value]


@register_oat_distribution(dist.Normal)
class NormalOAT(OATDistribution, dist.Normal):
    """Normal distribution with closed-form CRPS."""
    
    name = "normal"

    def crps(self, target: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Analytical CRPS for a Normal distribution.

        CRPS(N(μ,σ), y) = σ * [z(2Φ(z)−1) + 2φ(z) − 1/√π]  where z=(y−μ)/σ
        """
        std = self.scale
        mean = self.loc
        std_normal = dist.Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = (target - mean) / std
        Phi = std_normal.cdf(z)
        phi = torch.exp(std_normal.log_prob(z))
        inv_sqrt_pi = 1.0 / torch.sqrt(torch.tensor(torch.pi, device=std.device, dtype=std.dtype))
        return std * (z * (2 * Phi - 1) + 2 * phi - inv_sqrt_pi)
    
    def innovation(self, target: torch.Tensor) -> torch.Tensor:
        """For a Normal distribution, the innovation is simply the standardized residual."""
        return (target - self.loc) / self.scale


@register_oat_distribution(dist.Laplace)
class LaplaceOAT(OATDistribution, dist.Laplace):
    """Laplace distribution with closed-form CRPS."""
    
    name = "laplace"

    def crps(self, target: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """CRPS(Laplace(μ, b), y) = |y−μ| + b·exp(−|y−μ|/b) − 3b/4"""
        b = self.scale
        diff = torch.abs(target - self.loc)
        return diff + b * torch.exp(-diff / b) - 0.75 * b


@register_oat_distribution(dist.Exponential)
class ExponentialOAT(OATDistribution, dist.Exponential):
    """Exponential distribution with closed-form CRPS."""
    
    name = "exponential"

    def crps(self, target: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """CRPS(Exp(λ), y) = y − 3/(2λ) + (2/λ)·exp(−λy)   [y ≥ 0]"""
        lam = self.rate
        return target - 1.5 / lam + (2.0 / lam) * torch.exp(-lam * target)


@register_oat_distribution(dist.LogNormal)
class LogNormalOAT(OATDistribution, dist.LogNormal):
    """Log-Normal distribution with closed-form CRPS."""
    
    name = "lognormal"

    def crps(self, target: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """CRPS(LogNormal(μ, σ), y) = y·(2Φ(z)−1) − 2m·(Φ(z−σ) − Φ(−σ/√2))
        where z = (log y − μ)/σ, m = exp(μ + σ²/2).
        """
        mu, sigma = self.loc, self.scale
        
        #TODO consider a safety clamp
        m = torch.exp(mu + 0.5 * sigma ** 2)
        # TODO if applied on normalized space, target needs to be correctly transformer to preserve positivity
        # consider clamp? 
        z = (torch.log(target) - mu) / sigma 
        std_normal = dist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        return (
            target * (2 * std_normal.cdf(z) - 1)
            - 2 * m * (std_normal.cdf(z - sigma) - std_normal.cdf(-sigma / math.sqrt(2)))
        )

@register_oat_distribution(None)  # placeholder for custom distribution without a base torch class
class ZeroInflatedLogNormal(OATDistribution, torch.distributions.Distribution):
    """Zero-inflated LogNormal: P(X=0) = gate, P(X ~ LogN(loc, scale)) = 1 - gate.

    Designed for non-negative variables with excess zeros (e.g. precipitation,
    wind gusts). The analytical CRPS gives full gradients through all three
    parameters even though the Bernoulli gate is not reparameterizable.
    """
    
    name = "zero_inflated_lognormal"

    arg_constraints = {
        "gate": constraints.unit_interval,
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.nonnegative
    has_rsample = False  # Bernoulli gate is not reparameterizable

    def __init__(
        self,
        gate: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        validate_args=None,
    ):
        self.gate = gate
        self.loc = loc
        self.scale = scale
        batch_shape = torch.broadcast_shapes(gate.shape, loc.shape, scale.shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(value.dtype).eps
        gate = self.gate.clamp(min=eps, max=1 - eps)  # Avoid log(0) issues
        log_gate = torch.log(gate)
        log_not_gate = torch.log((1 - gate))
        # Clamp value before passing to LogNormal to avoid log(0) = -inf in the
        # false branch of torch.where (both branches are evaluated eagerly).
        lognormal = dist.LogNormal(self.loc, self.scale)
        return torch.where(
            value == 0,
            log_gate,
            log_not_gate + lognormal.log_prob(value.clamp(min=eps)),
        )
        
    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(value.dtype).eps
        lognormal = dist.LogNormal(self.loc, self.scale)
        return torch.where(
            value == 0,
            self.gate,
            self.gate + (1 - self.gate) * lognormal.cdf(value.clamp(min=eps)),
        )

    @property
    def mean(self) -> torch.Tensor:
        """E[X] = (1 - gate) * exp(loc + scale² / 2)."""
        return (1 - self.gate) * torch.exp(self.loc + 0.5 * self.scale ** 2)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            is_zero = torch.bernoulli(self.gate.expand(shape)).bool()
            lognormal_samples = dist.LogNormal(self.loc, self.scale).sample(sample_shape)
            return torch.where(is_zero, torch.zeros_like(lognormal_samples), lognormal_samples)

    def crps(self, target: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        # Verified numerically against a clean Monte-Carlo estimate in
        # test/oat_test/model_module/test_distributions.py::test_ziln_crps_vs_montecarlo
        # (agreement < 3e-3 across gate/loc/scale regimes and targets incl. 0).
        # Derivation (for target >= 0, P = gate*δ₀ + (1-gate)*LogN(loc, scale)):
        #   CRPS = gate*target
        #        + (1-gate)*CRPS(LogN(loc,scale), target)
        #        + 2*gate*(1-gate)*m*(Φ(σ/√2) - 1)
        # where m = exp(loc + scale²/2) and Φ is the standard normal CDF.
        # The cross-term 2*gate*(1-gate)*m*(Φ(σ/√2)-1) is always ≤ 0, reflecting
        # the sharpening effect of the point mass at zero.
        # At target=0 the formula simplifies to 2*m*(1-gate)²*Φ(−σ/√2), avoiding
        # the log(0) singularity in the LogNormal branch.
        pi = self.gate
        mu, sigma = self.loc, self.scale
        m = torch.exp(mu + 0.5 * sigma ** 2)
        std_normal = dist.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        eps = torch.finfo(target.dtype).eps
        z = (torch.log(target.clamp(min=eps)) - mu) / sigma
        lognormal_crps = (
            target * (2 * std_normal.cdf(z) - 1)
            - 2 * m * (std_normal.cdf(z - sigma) - std_normal.cdf(-sigma / math.sqrt(2)))
        )
        cross_term = 2 * pi * (1 - pi) * m * (std_normal.cdf(sigma / math.sqrt(2)) - 1)
        crps_at_pos = pi * target + (1 - pi) * lognormal_crps + cross_term

        crps_at_zero = 2 * m * (1 - pi) ** 2 * std_normal.cdf(-sigma / math.sqrt(2))
        return torch.where(target == 0, crps_at_zero, crps_at_pos)


class DistributionMap(torch.nn.Module):

    def __init__(self,
                 dist_class: type[Distribution],
                 in_features: int,
                 param_order: list[str] | None = None):
        """
        Initialize a distribution layer.
        Args:
            dist_class: A PyTorch distribution class that inherits from torch.distributions.Distribution.
            in_features: The number of input features for the linear layer that generates distribution parameters.
        Raises:
            AssertionError: If the distribution has multivariate parameters (event_dim != 0), which are not supported.
        Attributes:
            dist_class: The distribution class (wrapped as OATDistribution subclass).
            support: The support of the distribution.
            num_params: The number of parameters required by the distribution.
            net: A linear layer that maps from in_features to num_params.
        """
        check_if_valid_distribution(dist_class)
        super().__init__()
        self.dist_class = as_oat_distribution(dist_class)
        self.support = dist_class.support
        self.num_params = len(dist_class.arg_constraints)
        self.net = torch.nn.Linear(in_features, self.num_params)
        if param_order is not None:
            unknown = set(param_order) - set(dist_class.arg_constraints.keys())
            if len(unknown) > 0:
                raise ValueError(f"Unknown parameters in param_order: {unknown}")
            self.param_names = param_order
        else:
            self.param_names = list(dist_class.arg_constraints.keys())

    def forward(self, input: torch.Tensor) -> OATDistribution:
        """
        Generate a probability distribution from input data.
        This method processes input through a neural network and transforms the outputs
        into valid distribution parameters according to the constraints defined by the
        distribution class.
        Args:
            input (torch.Tensor): Input tensor to be processed by the network.
        Returns:
            OATDistribution: A probability distribution object initialized with the
                transformed network outputs as parameters.
        Notes:
            - The network output is transformed for each distribution parameter using
              the appropriate constraint transformations.
            - The number of output dimensions must match the number of distribution
              parameters.
        """
        dist_params = {}
        out = self.net(input)
        for i, name in enumerate(self.param_names):
            constraint = self.dist_class.arg_constraints[name]
            dist_params[name] = transform_to(constraint)(out[..., i])
        return self.dist_class(**dist_params)
