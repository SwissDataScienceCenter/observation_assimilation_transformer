import pytest
import torch
from torch import distributions as dist
from torch.distributions.constraint_registry import transform_to

from oat.model_module.distributions import (
    DistributionMap,
    ExponentialOAT,
    LaplaceOAT,
    LogNormalOAT,
    NormalOAT,
    OATDistribution,
    ZeroInflatedLogNormal,
    as_oat_distribution,
    check_if_valid_distribution,
    register_oat_distribution,
)


# ── check_if_valid_distribution ───────────────────────────────────────────────

def test_check_if_valid_distribution_accepts_univariate() -> None:
    check_if_valid_distribution(dist.Normal)  # should not raise


def test_check_if_valid_distribution_rejects_multivariate() -> None:
    with pytest.raises(ValueError):
        check_if_valid_distribution(dist.MultivariateNormal)


def test_check_if_valid_distribution_rejects_discrete() -> None:
    with pytest.raises(ValueError, match="rsample"):
        check_if_valid_distribution(dist.Poisson)


# ── as_oat_distribution ───────────────────────────────────────────────────────

def test_as_oat_distribution_wraps_plain_distribution() -> None:
    wrapped = as_oat_distribution(dist.Normal)
    assert issubclass(wrapped, OATDistribution)
    assert issubclass(wrapped, dist.Normal)


def test_as_oat_distribution_noop_for_subclass() -> None:
    assert as_oat_distribution(NormalOAT) is NormalOAT


def test_as_oat_distribution_uses_registry_for_normal() -> None:
    assert as_oat_distribution(dist.Normal) is NormalOAT


def test_distribution_map_normal_returns_normal_oat_instance() -> None:
    layer = DistributionMap(dist.Normal, in_features=8)
    x = torch.randn(4, 10, 8)
    d = layer(x)
    assert isinstance(d, NormalOAT)


# ── OATDistribution.crps — MC path ───────────────────────────────────────────

def test_oat_distribution_mc_crps_shape() -> None:
    B, S = 4, 10
    GammaCRPS = as_oat_distribution(dist.Gamma)
    d = GammaCRPS(concentration=torch.ones(B, S), rate=torch.ones(B, S))
    target = d.sample()
    crps = d.crps(target, num_samples=20)
    assert crps.shape == torch.Size([B, S])


def test_oat_distribution_mc_crps_non_negative() -> None:
    B, S = 4, 10
    GammaCRPS = as_oat_distribution(dist.Gamma)
    d = GammaCRPS(concentration=torch.ones(B, S) * 2, rate=torch.ones(B, S))
    target = d.sample()
    crps = d.crps(target, num_samples=200)
    assert (crps >= 0).all()


# ── NormalOAT.crps — analytical path ─────────────────────────────────────────

def test_normal_oat_crps_shape() -> None:
    B, S = 4, 10
    d = NormalOAT(loc=torch.zeros(B, S), scale=torch.ones(B, S))
    target = torch.randn(B, S)
    crps = d.crps(target)
    assert crps.shape == torch.Size([B, S])


def test_normal_oat_crps_non_negative() -> None:
    B, S = 8, 20
    d = NormalOAT(loc=torch.zeros(B, S), scale=torch.ones(B, S))
    target = torch.randn(B, S)
    crps = d.crps(target)
    assert (crps >= 0).all()


def test_normal_oat_analytical_close_to_mc() -> None:
    """Analytical and MC CRPS should agree to within sampling noise."""
    torch.manual_seed(42)
    B, S = 8, 20
    loc = torch.randn(B, S)
    scale = torch.rand(B, S) + 0.5
    d = NormalOAT(loc=loc, scale=scale)
    target = torch.randn(B, S)

    analytical = d.crps(target)
    mc = OATDistribution.crps(d, target, num_samples=2000)

    assert torch.allclose(analytical, mc, atol=0.1), (
        f"Max deviation: {(analytical - mc).abs().max().item():.4f}"
    )


def test_normal_oat_uses_registered_class() -> None:
    assert as_oat_distribution(dist.Normal) is NormalOAT


def test_normal_oat_is_oat_distribution() -> None:
    assert issubclass(NormalOAT, OATDistribution)
    assert issubclass(NormalOAT, dist.Normal)


# ── Analytical distributions — shape, non-negativity, MC agreement ───────────

def _make_laplace() -> tuple[str, OATDistribution, torch.Tensor]:
    torch.manual_seed(0)
    B, S = 4, 10
    d = LaplaceOAT(loc=torch.randn(B, S), scale=torch.rand(B, S) + 0.5)
    return "Laplace", d, torch.randn(B, S)


def _make_exponential() -> tuple[str, OATDistribution, torch.Tensor]:
    torch.manual_seed(1)
    B, S = 4, 10
    d = ExponentialOAT(rate=torch.rand(B, S) + 0.5)
    return "Exponential", d, d.sample()  # sample guarantees y ≥ 0


def _make_lognormal() -> tuple[str, OATDistribution, torch.Tensor]:
    torch.manual_seed(2)
    B, S = 4, 10
    d = LogNormalOAT(loc=torch.randn(B, S), scale=torch.rand(B, S) + 0.3)
    return "LogNormal", d, d.sample()


@pytest.mark.parametrize(
    "name,d,target",
    [_make_laplace(), _make_exponential(), _make_lognormal()],
    ids=["Laplace", "Exponential", "LogNormal"],
)
def test_analytical_crps_shape(name: str, d: OATDistribution, target: torch.Tensor) -> None:
    crps = d.crps(target)
    assert crps.shape == target.shape, f"{name}: expected shape {target.shape}, got {crps.shape}"


@pytest.mark.parametrize(
    "name,d,target",
    [_make_laplace(), _make_exponential(), _make_lognormal()],
    ids=["Laplace", "Exponential", "LogNormal"],
)
def test_analytical_crps_non_negative(name: str, d: OATDistribution, target: torch.Tensor) -> None:
    crps = d.crps(target)
    assert (crps >= 0).all(), f"{name}: CRPS has negative values: {crps.min().item():.4f}"


@pytest.mark.parametrize(
    "name,d,target",
    [_make_laplace(), _make_exponential(), _make_lognormal()],
    ids=["Laplace", "Exponential", "LogNormal"],
)
def test_analytical_crps_close_to_mc(name: str, d: OATDistribution, target: torch.Tensor) -> None:
    torch.manual_seed(42)
    analytical = d.crps(target)
    mc = OATDistribution.crps(d, target, num_samples=5000)
    max_dev = (analytical - mc).abs().max().item()
    assert torch.allclose(analytical, mc, atol=0.1), (
        f"{name}: max deviation {max_dev:.4f}"
    )


def test_distribution_map_uses_registered_classes() -> None:
    for base, expected_cls in [
        (dist.Normal, NormalOAT),
        (dist.Laplace, LaplaceOAT),
        (dist.Exponential, ExponentialOAT),
        (dist.LogNormal, LogNormalOAT),
    ]:
        layer = DistributionMap(base, in_features=8)
        d = layer(torch.randn(2, 5, 8))
        assert isinstance(d, expected_cls), f"{base.__name__}: got {type(d).__name__}"


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


def test_distribution_map_init_rejects_discrete() -> None:
    with pytest.raises(ValueError, match="rsample"):
        DistributionMap(dist.Poisson, in_features=16)


# ── DistributionMap.forward ───────────────────────────────────────────────────

def test_distribution_map_returns_oat_distribution() -> None:
    layer = DistributionMap(dist.Normal, in_features=8)
    x = torch.randn(4, 10, 8)
    d = layer(x)
    assert isinstance(d, OATDistribution)
    assert hasattr(d, "crps")


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


def test_distribution_map_crps_shape() -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)
    target = torch.randn(batch, stations)

    crps = d.crps(target, num_samples=20)

    assert crps.shape == torch.Size([batch, stations])


# ── Parametrised over all distributions from the notebook ────────────────────

_VALID_DISTRIBUTIONS = [
    dist.LogNormal,
    dist.Exponential,
    dist.Weibull,
    dist.Laplace,
    dist.Gumbel,
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
    assert isinstance(d, OATDistribution)
    assert d.batch_shape == torch.Size([batch, stations])

    with torch.no_grad():
        target = d.sample()
    assert target.shape == torch.Size([batch, stations])

    log_probs = d.log_prob(target)
    assert log_probs.shape == torch.Size([batch, stations])
    assert torch.isfinite(log_probs).all()

    samples = d.rsample(sample_shape=(5,))
    assert samples.shape == torch.Size([5, batch, stations])

    crps = d.crps(target, num_samples=10)
    assert crps.shape == torch.Size([batch, stations])


# ── ZeroInflatedLogNormal ─────────────────────────────────────────────────────

def test_ziln_sample_shape() -> None:
    B, S = 4, 10
    d = ZeroInflatedLogNormal(gate=torch.full((B, S), 0.3), loc=torch.zeros(B, S), scale=torch.ones(B, S))
    samples = d.sample((7,))
    assert samples.shape == torch.Size([7, B, S])


def test_ziln_sample_no_args_returns_batch_shape() -> None:
    B, S = 4, 10
    d = ZeroInflatedLogNormal(gate=torch.full((B, S), 0.3), loc=torch.zeros(B, S), scale=torch.ones(B, S))
    samples = d.sample()
    assert samples.shape == torch.Size([B, S])


def test_ziln_sample_non_negative() -> None:
    torch.manual_seed(0)
    B, S = 4, 10
    d = ZeroInflatedLogNormal(gate=torch.rand(B, S), loc=torch.randn(B, S), scale=torch.rand(B, S) + 0.3)
    samples = d.sample((200,))
    assert (samples >= 0).all()
    
    
def test_ziln_log_prob_zero_in_support() -> None:
    """log_prob should not return -inf for zero values (which are in the support)."""
    torch.manual_seed(0)
    B, S = 4, 10
    d = ZeroInflatedLogNormal(gate=torch.rand(B, S), loc=torch.randn(B, S), scale=torch.rand(B, S) + 0.3)
    zeros = torch.zeros(B, S)
    log_probs = d.log_prob(zeros)
    assert log_probs.shape == torch.Size([B, S])
    assert torch.isfinite(log_probs).all(), "log_prob returns -inf for zero values"
    assert torch.allclose(log_probs, torch.log(d.gate)), "log_prob for zero values should equal log(gate)"


def test_ziln_sample_zero_fraction_matches_gate() -> None:
    """Empirical fraction of zeros should be close to gate (large N, loose tol)."""
    torch.manual_seed(42)
    gate_val = 0.6
    d = ZeroInflatedLogNormal(
        gate=torch.tensor([[gate_val]]),
        loc=torch.zeros(1, 1),
        scale=torch.ones(1, 1),
    )
    N = 5000
    samples = d.sample((N,))  # [N, 1, 1]
    empirical = (samples == 0).float().mean().item()
    assert abs(empirical - gate_val) < 0.05, f"Expected ~{gate_val}, got {empirical:.3f}"


def test_ziln_sample_gate_zero_no_zeros() -> None:
    """gate=0 means no zeros."""
    torch.manual_seed(0)
    d = ZeroInflatedLogNormal(gate=torch.zeros(4, 10), loc=torch.zeros(4, 10), scale=torch.ones(4, 10))
    samples = d.sample((50,))
    assert (samples > 0).all()


def test_ziln_sample_gate_one_all_zeros() -> None:
    """gate=1 means all zeros."""
    d = ZeroInflatedLogNormal(gate=torch.ones(4, 10), loc=torch.zeros(4, 10), scale=torch.ones(4, 10))
    samples = d.sample((50,))
    assert (samples == 0).all()


def test_ziln_crps_gradients_all_params() -> None:
    """CRPS must back-propagate through gate, loc, and scale."""
    torch.manual_seed(0)
    B, S = 4, 10
    gate = torch.rand(B, S).requires_grad_(True)
    loc = torch.randn(B, S).requires_grad_(True)
    scale = torch.rand(B, S).add(0.3).requires_grad_(True)
    d = ZeroInflatedLogNormal(gate=gate, loc=loc, scale=scale)
    with torch.no_grad():
        target = d.sample()

    d.crps(target).sum().backward()

    for param, name in [(gate, "gate"), (loc, "loc"), (scale, "scale")]:
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


# ── Gradient tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "name,make_d,param_names",
    [
        ("NormalOAT", lambda: NormalOAT(
            loc=torch.randn(4, 10, requires_grad=True),
            scale=torch.rand(4, 10).add(0.5).requires_grad_(True),
        ), ["loc", "scale"]),
        ("LaplaceOAT", lambda: LaplaceOAT(
            loc=torch.randn(4, 10, requires_grad=True),
            scale=torch.rand(4, 10).add(0.5).requires_grad_(True),
        ), ["loc", "scale"]),
        ("ExponentialOAT", lambda: ExponentialOAT(
            rate=torch.rand(4, 10).add(0.5).requires_grad_(True),
        ), ["rate"]),
        ("LogNormalOAT", lambda: LogNormalOAT(
            loc=torch.randn(4, 10, requires_grad=True),
            scale=torch.rand(4, 10).add(0.3).requires_grad_(True),
        ), ["loc", "scale"]),
    ],
    ids=["NormalOAT", "LaplaceOAT", "ExponentialOAT", "LogNormalOAT"],
)
def test_analytical_crps_gradients(name: str, make_d, param_names: list[str]) -> None:
    torch.manual_seed(0)
    d = make_d()
    params = [getattr(d, p) for p in param_names]
    with torch.no_grad():
        target = d.sample()

    crps = d.crps(target)
    crps.sum().backward()

    for p, pname in zip(params, param_names):
        assert p.grad is not None, f"{name}: no gradient for {pname}"
        assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient for {pname}"


def test_gamma_mc_crps_gradients_wrt_all_params() -> None:
    """MC-CRPS via rsample is differentiable w.r.t. both Gamma parameters."""
    torch.manual_seed(0)
    GammaOAT = as_oat_distribution(dist.Gamma)
    concentration = torch.rand(4, 10).add(1.0).requires_grad_(True)
    rate = torch.rand(4, 10).add(0.5).requires_grad_(True)
    d = GammaOAT(concentration=concentration, rate=rate)
    with torch.no_grad():
        target = d.sample()
    d.crps(target).sum().backward()
    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()
    assert rate.grad is not None
    assert torch.isfinite(rate.grad).all()


def test_mc_crps_gradients_through_rsample() -> None:
    """MC CRPS must back-propagate through rsample for a non-analytical distribution."""
    torch.manual_seed(0)
    concentration = torch.rand(4, 10).add(1.0).requires_grad_(True)
    rate = torch.rand(4, 10).add(0.5).requires_grad_(True)
    GammaMC = as_oat_distribution(dist.Weibull)  # no analytical CRPS → uses MC path
    d = GammaMC(scale=concentration, concentration=rate)
    with torch.no_grad():
        target = d.sample()

    crps = OATDistribution.crps(d, target, num_samples=50)
    crps.sum().backward()

    assert concentration.grad is not None
    assert torch.isfinite(concentration.grad).all()
    assert rate.grad is not None
    assert torch.isfinite(rate.grad).all()


def test_distribution_map_crps_gradients_through_network() -> None:
    """CRPS must flow gradients back through the DistributionMap linear layer."""
    torch.manual_seed(0)
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist.Normal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)
    with torch.no_grad():
        target = d.sample()

    crps = d.crps(target)
    crps.sum().backward()

    assert layer.net.weight.grad is not None
    assert torch.isfinite(layer.net.weight.grad).all()
    assert layer.net.bias.grad is not None
    assert torch.isfinite(layer.net.bias.grad).all()


# ── Innovation tests ─────────────────────────────────────────────────────────

def test_normal_oat_innovation_matches_standardized_residual() -> None:
    B, S = 4, 10
    loc = torch.randn(B, S)
    scale = torch.rand(B, S) + 0.5
    d = NormalOAT(loc=loc, scale=scale)
    target = torch.randn(B, S)

    innovation = d.innovation(target)
    expected = (target - loc) / scale

    assert innovation.shape == torch.Size([B, S])
    assert torch.allclose(innovation, expected)


def _safe_innovation_target(d: OATDistribution) -> torch.Tensor:
    """Pick an in-support target that keeps CDF away from 0/1 tails."""
    with torch.no_grad():
        samples = d.sample((256,))
    return samples.median(dim=0).values


def _moderate_tail_targets(d: OATDistribution) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick lower/upper tail targets that should remain finite."""
    with torch.no_grad():
        samples = d.sample((1024,))
    return samples.quantile(0.1, dim=0), samples.quantile(0.9, dim=0)


def _extreme_tail_targets(d: OATDistribution) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick lower/upper tail targets that may saturate to +-inf in innovation()."""
    with torch.no_grad():
        samples = d.sample((4096,))
    return samples.quantile(0.01, dim=0), samples.quantile(0.99, dim=0)


@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_registered_innovation_shape_and_finite(dist_class: type) -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    target = _safe_innovation_target(d)
    innovation = d.innovation(target)

    assert innovation.shape == torch.Size([batch, stations])
    assert torch.isfinite(innovation).all()


@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_registered_innovation_gradients_through_network(
    dist_class: type,
) -> None:
    """Innovation must back-propagate through DistributionMap.net for registered classes."""
    torch.manual_seed(0)
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    target = _safe_innovation_target(d)
    innovation = d.innovation(target)
    loss = innovation.square().mean()
    loss.backward()

    assert layer.net.weight.grad is not None
    assert torch.isfinite(layer.net.weight.grad).all()
    assert layer.net.bias.grad is not None
    assert torch.isfinite(layer.net.bias.grad).all()


@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_registered_innovation_moderate_tails_are_finite(
    dist_class: type,
) -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    lower_target, upper_target = _moderate_tail_targets(d)
    lower_innovation = d.innovation(lower_target)
    upper_innovation = d.innovation(upper_target)

    assert torch.isfinite(lower_innovation).all()
    assert torch.isfinite(upper_innovation).all()
    assert (lower_innovation < upper_innovation).all()

    assert (upper_innovation - lower_innovation > 0).all()


@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_registered_innovation_extreme_tails_behavior(
    dist_class: type,
) -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)

    lower_target, upper_target = _extreme_tail_targets(d)
    lower_innovation = d.innovation(lower_target)
    upper_innovation = d.innovation(upper_target)

    assert torch.isfinite(lower_innovation).all()
    assert torch.isfinite(upper_innovation).all()
    assert (lower_innovation < upper_innovation).all()

    assert (upper_innovation - lower_innovation > 0).all()

@pytest.mark.parametrize("dist_class", _VALID_DISTRIBUTIONS, ids=lambda d: d.__name__)
def test_distribution_map_allows_gradients(dist_class: type) -> None:
    batch, stations, in_features = 4, 10, 8
    layer = DistributionMap(dist_class, in_features=in_features)
    x = torch.randn(batch, stations, in_features, requires_grad=True)

    d = layer(x)

    with torch.no_grad():
        # get a sample in the correct support (no gradients)
        target = d.sample()
    # get an output that is differentiable wrt to x
    log_probs = d.log_prob(target)

    assert x.grad is None
    log_probs.sum().backward() # sum needed to make it a scalar (no need for grad_in input)
    assert x.grad is not None
    
    # check the same from samples obtained from d
    x = torch.randn(batch, stations, in_features, requires_grad=True)
    d = layer(x)
    if d.has_rsample:
        d.rsample(sample_shape=(5,)).sum().backward()
        assert x.grad is not None
        
        
def test_ziln_crps_vs_montecarlo():
    
    torch.manual_seed(0)
    batch, stations, in_features = 4, 10, 8
    
    layer = DistributionMap(ZeroInflatedLogNormal, in_features=in_features)
    x = torch.randn(batch, stations, in_features)
    d = layer(x)
    
    from oat.predictor_module.functional_metrics import mc_crps
    
    # [100,4,10]
    samples = d.sample((5000,))
    samples = samples.permute(1,2,0)  # [4,10,100]
    samples = samples.unsqueeze(-1)  # [4,10,100,1]
    # put some true zeros in the samples to make sure the zero-inflation is tested 
    # at random locations
    zero_mask = torch.rand_like(samples) < 0.1
    samples = torch.where(zero_mask, torch.zeros_like(samples), samples) 
    
    
    target = d.sample()  # [4,10]
    target = target.unsqueeze(-1)  # [4,10,1]
    # put true zeros in the target to make sure the zero-inflation is tested at random locations
    zero_mask_target = torch.rand_like(target) < 0.1
    target = torch.where(zero_mask_target, torch.zeros_like(target), target)
    
    mask = torch.ones_like(target)  # [4,10,1]
    
    
    crps_estimate = mc_crps(y_target=target, y_hat=samples, mask=mask, aggregation="variable")
    
    closed_crps = d.crps(target.squeeze(-1)).mean()
    
    
    # check the MC estimate is close to the closed-form CRPS, up to the first comma
    assert torch.allclose(crps_estimate, closed_crps, atol=0.01), (
        f"MC CRPS: {crps_estimate}, Closed-form CRPS: {closed_crps}"
    )
    

# TODO add the other CRPS tests for the other distributions, similar to the above test for ZILN