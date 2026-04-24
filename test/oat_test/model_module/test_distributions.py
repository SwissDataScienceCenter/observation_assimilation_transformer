import pytest
import torch
from torch import distributions as dist
from torch.distributions.constraint_registry import transform_to

from oat.model_module.distributions import DistributionMap, check_if_valid_distribution


# ── check_if_valid_distribution ───────────────────────────────────────────────

def test_check_if_valid_distribution_accepts_univariate() -> None:
    # Normal has loc (real) and scale (positive) — both event_dim == 0
    check_if_valid_distribution(dist.Normal)  # should not raise


def test_check_if_valid_distribution_rejects_multivariate() -> None:
    # MultivariateNormal has covariance_matrix with event_dim > 0
    with pytest.raises(ValueError):
        check_if_valid_distribution(dist.MultivariateNormal)


# ── DistributionMap.__init__ ──────────────────────────────────────────────────

def test_distribution_map_init_attributes() -> None:
    in_features = 16
    layer = DistributionMap(dist.Normal, in_features=in_features)

    assert layer.num_params == 2  # loc and scale
    assert isinstance(layer.net, torch.nn.Linear)
    assert layer.net.in_features == in_features
    assert layer.net.out_features == 2
    assert layer.support is dist.Normal.support


def test_distribution_map_init_rejects_multivariate() -> None:
    with pytest.raises(ValueError):
        DistributionMap(dist.MultivariateNormal, in_features=16)


# ── DistributionMap.forward ───────────────────────────────────────────────────

def test_distribution_map_forward_returns_distribution_with_correct_batch_shape() -> None:
    batch, stations, in_features = 32, 100, 8
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)

    d = layer(x)

    assert isinstance(d, dist.Normal)
    assert d.batch_shape == torch.Size([batch, stations])


def test_distribution_map_forward_constraints_respected_normal() -> None:
    batch, stations, in_features = 32, 100, 8
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)

    d = layer(x)

    # scale (std) must be strictly positive after transform_to(positive)
    assert (d.scale > 0).all()


def test_distribution_map_forward_constraints_respected_gamma() -> None:
    batch, stations, in_features = 32, 100, 8
    layer = DistributionMap(dist.Gamma, in_features=in_features)
    x = torch.randn(batch, stations, in_features)

    d = layer(x)

    assert (d.concentration > 0).all()
    assert (d.rate > 0).all()


def test_distribution_map_forward_log_prob_shape() -> None:
    batch, stations, in_features = 32, 100, 8
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    # Map unconstrained values into the distribution's support (mirrors notebook usage)
    target = transform_to(d.support)(torch.randn(batch, stations))
    log_probs = d.log_prob(target)

    assert log_probs.shape == torch.Size([batch, stations])
    assert torch.isfinite(log_probs).all()


def test_distribution_map_forward_rsample_shape() -> None:
    batch, stations, in_features = 32, 100, 8
    n_samples = 10
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    samples = d.rsample(sample_shape=(n_samples,))

    assert samples.shape == torch.Size([n_samples, batch, stations])


# ── Parametrised over all distributions from the notebook ────────────────────

_VALID_DISTRIBUTIONS = [
    dist.Poisson,
    dist.LogNormal,
    dist.Exponential,
    dist.Weibull,
    dist.Laplace,
    dist.HalfCauchy,
    dist.StudentT,
    dist.Gumbel,
    dist.InverseGamma,
    dist.Chi2,
    dist.Beta,
    dist.Gamma,
    dist.Normal,
    dist.Cauchy,
]


@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_all_valid_distributions(dist_class: type) -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features)

    d = layer(x)

    assert isinstance(d, dist_class)
    assert d.batch_shape == torch.Size([batch, stations])

    # Sample from the distribution to get values guaranteed to be in the support
    with torch.no_grad():
        target = d.sample()
    assert target.shape == torch.Size([batch, stations])

    log_probs = d.log_prob(target)
    assert log_probs.shape == torch.Size([batch, stations])
    assert torch.isfinite(log_probs).all()

    # Use rsample for continuous distributions, sample for discrete ones
    n_samples = 5
    if dist_class.has_rsample:
        samples = d.rsample(sample_shape=(n_samples,))
    else:
        samples = d.sample(sample_shape=(n_samples,))
    assert samples.shape == torch.Size([n_samples, batch, stations])
