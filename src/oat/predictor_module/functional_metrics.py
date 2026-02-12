from typing import Literal

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
def sample_mae(y_target, y_hat, mask=None, aggregation: Literal["all", "variable"] = "all"):
    """
    Mean Absolute Error over all samples.

    Args:
        y_target: [batch, station, feature]
        y_hat: [batch, station, sample, feature]
        mask: [batch, station, feature] or None
        aggregation: "all" -> single MAE, "variable" -> MAE per feature
    """

    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)

    if mask is None:
        mask = torch.ones_like(y_target)

    y_target_exp = y_target.unsqueeze(-2)   # [batch, station, 1, feature]
    mask_exp = mask.unsqueeze(-2).float()   # [batch, station, 1, feature]

    abs_errors = torch.abs(y_target_exp - y_hat) * mask_exp  # [batch, station, sample, feature]

    if aggregation == "all":
        # Sum over all dimensions
        num = abs_errors.sum()
        denom = mask_exp.sum() * y_hat.shape[-2]
        if denom.item() == 0:
            return torch.tensor(0.0, device=y_hat.device)
        return num / denom

    elif aggregation == "variable":
        # keep feature dimension
        num = abs_errors.sum(dim=[0, 1, 2])      # [feature]
        denom = (mask_exp.sum(dim=[0, 1, 2]) * y_hat.shape[-2])  # [feature]

        return torch.where(denom > 0, num / denom, torch.zeros_like(num))

    raise ValueError(f"Invalid aggregation: {aggregation}")

@register_metric()
def point_mae(y_target, y_hat, mask=None, aggregation: Literal["all", "variable"] = "all"):
    """
    Mean Absolute Error using the median of samples.

    Args:
        y_target: [batch, station, feature]
        y_hat: [batch, station, sample, feature]
        mask: [batch, station, feature] or None
        aggregation: "all" -> single MAE, "variable" -> MAE per feature
    """
    
    # Ensure shapes match
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)

    if mask is None:
        mask = torch.ones_like(y_target)

    # Take median over samples
    median = torch.median(y_hat, dim=-2).values

    abs_errors = torch.abs(y_target - median) * mask.float()

    if aggregation == "all":
        num = abs_errors.sum()
        denom = mask.sum()
        if denom.item() == 0:
            return torch.tensor(0.0, device=y_hat.device)
        return num / denom

    elif aggregation == "variable":
        num = abs_errors.sum(dim=[0,1])
        denom = mask.sum(dim=[0,1])

        return torch.where(denom > 0, num / denom, torch.zeros_like(num))

    raise ValueError(f"Invalid aggregation: {aggregation}")

@register_metric()
def mc_crps(y_target, y_hat, mask=None, aggregation: Literal["all", "variable"] = "all"):
    """
    Computes the Monte Carlo estimation of CRPS. 
    If aggregation = "all", this function effectively computes the Energy Score (multivariate CRPS)
    """

    # y_target.shape = [batch,station,features]
    # y_hat.shape = [batch,station,samples, features]
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)
    
    mc_samples = y_hat.shape[-2]
    if aggregation == "all":
        mask_exp = mask.unsqueeze(-2)
        
        # First term E(||X-y||)
        smpls_to_tgt_dist = y_hat - y_target.unsqueeze(-2) * mask_exp
        eucl_distance = torch.norm(smpls_to_tgt_dist, dim=-1)
        first_term = eucl_distance.mean(dim=-1)
        
        # Second term: E(||X-X'||)
        xj = y_hat.unsqueeze(-2)
        xi = y_hat.unsqueeze(-3)
        
        
        smpls_diff = (xi - xj) * mask.unsqueeze(-2).unsqueeze(-2)
        smpls_to_smpls_dist = torch.norm(smpls_diff, dim=-1)
        # divide by n*(n-1) because the diagonal of smpls_to_smpls_dist is 0
        second_term = smpls_to_smpls_dist.sum(dim=(-1,-2)) /  (mc_samples * (mc_samples -1))

        # E(||X-y||) - 0.5 E(||X-X'||)
        # TODO check this normalization
        # valid is valid if at least one feature was preset
        valid_locations = (mask.sum(dim=-1) > 0).float()
        score = first_term - 0.5 * second_term
        return (score * valid_locations).sum() / valid_locations.sum()

    elif aggregation == "variable":
        mask_exp = mask.unsqueeze(-2)

        # First term E(||X-y||)
        smpls_to_tgt_abs_diff = torch.abs(y_hat - y_target.unsqueeze(-2)) * mask_exp
        first_term = smpls_to_tgt_abs_diff.mean(dim=-2)
        
        # Second term: E(||X-X'||)
        xj = y_hat.unsqueeze(-2)
        xi = y_hat.unsqueeze(-3)
        
        smpls_to_smpls_abs_diff = torch.abs(xi - xj) * mask.unsqueeze(-2).unsqueeze(-2)
        
        second_term = smpls_to_smpls_abs_diff.sum(dim=(-3, -2)) / (mc_samples * (mc_samples - 1))
        
        score = first_term - 0.5 * second_term 
        masked_score = score * mask
        return masked_score.sum(dim=(0,1)) / mask.sum(dim=(0,1))
        # return (first_term - 0.5 * second_term).mean(dim=(0,1))