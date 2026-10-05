from oat.predictor_module.functional_metrics import METRICS

import torch
import pytest


@pytest.mark.parametrize(
    "metric_fn",
    list(METRICS.values()),
    ids=list(METRICS.keys()),
)
@pytest.mark.parametrize("aggregation", ['all', 'variable', 'station'])
def test_sample_mae(metric_fn: callable, aggregation: str) -> None:

    y_pred_samples = torch.randn(32,18,100,4)
    y_true = torch.randn(32,18,4)

    mask = torch.ones(32,18,4)
    mask_p = torch.rand_like(y_true)
    mask[mask_p < 0.2] = 0

    metrics = metric_fn(y_target=y_true, y_hat=y_pred_samples, mask=mask, aggregation=aggregation)

    if aggregation == 'all':
        assert metrics.ndim == 0
    elif aggregation == 'variable':
        assert metrics.ndim == 1
        assert metrics.shape[0] == 4
    elif aggregation == 'station':
        assert metrics.ndim == 2
        assert metrics.shape == (18, 4)

    # Test with samples and predictions all being different
    # by a scale of 2
    y_pred_samples = torch.ones(32,18,100,4)
    y_true = torch.ones(32,18,4) * 2

    metrics = metric_fn(y_target=y_true, y_hat=y_pred_samples, mask=mask, aggregation=aggregation)

    if aggregation == 'all':
        assert metrics.item() == 1.0
    else:
        assert (metrics == 1.0).all()

def test_point_mae_with_expanded_samples_matches_contiguous():
    """The NWP interpolator expands one value into its samples (stride 0).
    torch.median gives wrong values for such a tensor, so point_mae must
    give the same result as for a contiguous copy."""
    from oat.predictor_module.functional_metrics import point_mae

    torch.manual_seed(0)
    y_target = torch.randn(1, 50, 2) * 10 + 280
    y_pred = y_target + torch.randn(1, 50, 2)
    y_hat = y_pred.unsqueeze(-2).expand(-1, -1, 100, -1)
    expected = point_mae(y_target, y_hat.contiguous(), aggregation="station")
    torch.testing.assert_close(
        point_mae(y_target, y_hat, aggregation="station"), expected
    )
