import torch
import torch.distributions as td

from oat.baselines.ootg_likelihood import (
    DIST_PARAM_COUNTS,
    JointMixedDistribution,
    MixedVariableLikelihood,
    decoder_out_dim,
)


def _make_normal_only():
    return MixedVariableLikelihood({"temperature": "normal"})


def _make_mixed():
    return MixedVariableLikelihood(
        {"temperature": "normal", "precipitation": "zero_inflated_lognormal"}
    )


def test_dist_param_counts():
    assert DIST_PARAM_COUNTS["normal"] == 2
    assert DIST_PARAM_COUNTS["zero_inflated_lognormal"] == 3


def test_decoder_out_dim_normal():
    assert decoder_out_dim({"temperature": "normal"}) == 2


def test_decoder_out_dim_mixed():
    assert decoder_out_dim({"temperature": "normal", "precipitation": "zero_inflated_lognormal"}) == 5


def test_forward_normal_only_returns_joint():
    lik = _make_normal_only()
    x = torch.randn(2, 10, 2)
    d = lik(x)
    assert isinstance(d, JointMixedDistribution)


def test_forward_mixed_returns_joint():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5)
    d = lik(x)
    assert isinstance(d, JointMixedDistribution)


def test_log_prob_shape_normal():
    lik = _make_normal_only()
    x = torch.randn(2, 10, 2)
    d = lik(x)
    y = torch.randn(2, 10, 1)
    lp = d.log_prob(y)
    assert lp.shape == (2, 10, 1)


def test_log_prob_shape_mixed():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5)
    d = lik(x)
    # precipitation must be non-negative for ZILN
    y = torch.cat([torch.randn(2, 10, 1), torch.rand(2, 10, 1)], dim=-1)
    lp = d.log_prob(y)
    assert lp.shape == (2, 10, 2)


def test_log_prob_finite():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5)
    d = lik(x)
    y = torch.cat([torch.randn(2, 10, 1), torch.rand(2, 10, 1)], dim=-1)
    assert d.log_prob(y).isfinite().all()


def test_mean_shape_mixed():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5)
    d = lik(x)
    assert d.mean.shape == (2, 10, 2)


def test_sum_log_prob_loss():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5)
    d = lik(x)
    y = torch.cat([torch.randn(2, 10, 1), torch.rand(2, 10, 1)], dim=-1)
    loss = -d.log_prob(y).sum(-1).mean()
    assert loss.ndim == 0 and loss.isfinite()


def test_gradients_flow():
    lik = _make_mixed()
    x = torch.randn(2, 10, 5, requires_grad=True)
    d = lik(x)
    y = torch.cat([torch.randn(2, 10, 1), torch.rand(2, 10, 1)], dim=-1)
    loss = -d.log_prob(y).sum(-1).mean()
    loss.backward()
    assert x.grad is not None
