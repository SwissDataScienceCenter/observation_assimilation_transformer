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