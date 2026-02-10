from oat.predictor_module.functional_metrics import METRICS

import torch
import pytest


@pytest.mark.parametrize(
    "metric_fn",
    list(METRICS.values()),
    ids=list(METRICS.keys()),
)
def test_sample_mae(metric_fn):
    
    y_pred_samples = torch.randn(32,18,100,4)
    y_true = torch.randn(32,18,4)

    metrics = metric_fn(y_target=y_true, y_hat=y_pred_samples, mask=None)
    
    assert metrics.ndim == 0