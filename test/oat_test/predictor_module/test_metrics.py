from oat.predictor_module.functional_metrics import METRICS

import torch
import pytest


@pytest.mark.parametrize(
    "metric_fn",
    list(METRICS.values()),
    ids=list(METRICS.keys()),
)
@pytest.mark.parametrize("aggregation", ['all', 'variable'])
def test_sample_mae(metric_fn, aggregation):
    
    y_pred_samples = torch.randn(32,18,100,4)
    y_true = torch.randn(32,18,4)

    metrics = metric_fn(y_target=y_true, y_hat=y_pred_samples, mask=None, aggregation=aggregation)
    
    if aggregation == 'all':
        assert metrics.ndim == 0
    else:
        assert metrics.ndim == 1
        assert metrics.shape[0] == 4
        
        
    y_pred_samples = torch.ones(32,18,100,4)
    y_true = torch.ones(32,18,4) * 2

    metrics = metric_fn(y_target=y_true, y_hat=y_pred_samples, mask=None, aggregation=aggregation)
    
    if aggregation == 'all':
        assert metrics.item() == 1.0
    else:
        assert (metrics == 1.0).all()