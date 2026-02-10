import torch

METRICS = {}
def register_metric(name=None):
    def decorator(fn):
        metric_name = name or fn.__name__
        METRICS[metric_name] = fn
        return fn
    return decorator


def assert_prediction_matches_target(y_hat, y_target):
    assert y_hat.ndim == 4
    assert y_target.ndim == 3
    assert y_hat.shape[0] == y_target.shape[0], "Batch size mismatch"
    assert y_hat.shape[1] == y_target.shape[1], "Station size mismatch"
    assert y_hat.shape[-1] == y_target.shape[-1], "Feature size mismatch"

@register_metric()
def sample_mae(y_target, y_hat, mask):
    # y_hat_unorm shape: [batch,station,sample,feature]
    # target_obs shape: [batch,station,feature]
    # mask shape: [batch,station,feature]
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)
    
    if mask is None:
        mask = torch.ones_like(y_target)
    
    abs_errors = torch.abs(y_target.unsqueeze(-2) - y_hat)
    abs_errors *= mask.unsqueeze(-2)
    
    denom = (mask.unsqueeze(-2).sum() * y_hat.shape[-2])
    if denom == 0:
        return torch.tensor(0.0, device=y_hat.device)
    
    return abs_errors.sum() / denom

@register_metric()
def point_mae(y_target, y_hat, mask):
    # y_hat_unorm shape: [batch,station,sample,feature]
    # target_obs shape: [batch,station,feature]
    # mask shape: [batch,station,feature]
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)
    
    if mask is None:
        mask = torch.ones_like(y_target)
    
    median = torch.median(y_hat, dim=-2).values
    abs_errors = torch.abs(y_target - median)
    
    abs_errors *= mask
    
    denom = mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=y_hat.device)
    
    return abs_errors.sum() / denom
    
    
    