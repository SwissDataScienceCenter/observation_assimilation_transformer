from typing import Literal, Optional

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
def sample_mae(y_target, y_hat, mask=None, aggregation: Literal["all", "variable", "station"] = "all"):
    """
    Mean Absolute Error over all samples.

    Args:
        y_target: [batch, station, feature]
        y_hat: [batch, station, sample, feature]
        mask: [batch, station, feature] or None
        aggregation: "all" -> scalar, "variable" -> [feature], "station" -> [station, feature]
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

    elif aggregation == "station":
        num = abs_errors.sum(dim=[0, 2])                                    # [station, feature]
        denom = mask_exp.sum(dim=[0, 2]).clamp(min=1e-8) * y_hat.shape[-2]
        return num / denom

    raise ValueError(f"Invalid aggregation: {aggregation}")

@register_metric()
def point_mae(y_target, y_hat, mask=None, aggregation: Literal["all", "variable", "station"] = "all"):
    """
    Mean Absolute Error using the median of samples.

    Args:
        y_target: [batch, station, feature]
        y_hat: [batch, station, sample, feature]
        mask: [batch, station, feature] or None
        aggregation: "all" -> scalar, "variable" -> [feature], "station" -> [station, feature]
    """

    # Ensure shapes match
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)

    if mask is None:
        mask = torch.ones_like(y_target)

    # Take median over samples. contiguous(): torch.median gives wrong values
    # for an expanded (stride 0) tensor, e.g. the NWP interpolator samples.
    median = torch.median(y_hat.contiguous(), dim=-2).values

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

    elif aggregation == "station":
        num = abs_errors.sum(dim=0)                              # [station, feature]
        denom = mask.float().sum(dim=0).clamp(min=1e-8)
        return num / denom

    raise ValueError(f"Invalid aggregation: {aggregation}")

@register_metric()
def mc_crps(y_target: torch.Tensor, y_hat: torch.Tensor, mask: Optional[torch.Tensor] = None,
            aggregation: Literal["all", "variable", "station", "none"] = "all") -> torch.Tensor:
    """
    Computes the Monte Carlo estimation of CRPS. 
    If aggregation = "all", this function computes the average CRPS 
    across all variable, not the Energy Score.
    
    y_target.shape = [batch,station,features]
    y_hat.shape = [batch,station,samples, features]
    mask.shape = [batch,station,features]
    
    If aggregation = "all", output is a 0-dimensional tensor containing the averaged
    metric.
    If aggregation = "variable", output is 1-dimensional, with one value per variable.
    If aggregation = "none", output is the score for each element, shape
    [batch,station,features]. The caller must apply the mask.
    """
    
    if mask is None:
        mask = torch.ones_like(y_target)
        
    if mask.sum() == 0.0:
        raise ValueError("Mask does not contain any valid values.")

    
    assert_prediction_matches_target(y_hat=y_hat, y_target=y_target)
    
    mc_samples = y_hat.shape[-2]
    mask_exp = mask.unsqueeze(-2)

    # First term E(||X-y||)
    smpls_to_tgt_abs_diff = torch.abs(y_hat - y_target.unsqueeze(-2)) * mask_exp
    first_term = smpls_to_tgt_abs_diff.mean(dim=-2)
    
    # Second term: E(||X-X'||)
    xj = y_hat.unsqueeze(-2)
    xi = y_hat.unsqueeze(-3)
    
    smpls_to_smpls_abs_diff = torch.abs(xi - xj) * mask.unsqueeze(-2).unsqueeze(-2)
    
    # normalization is mc_samples * (mc_samples - 1) and not mc_samples**2
    # because the diagonal of smpls_to_smpls_abs_diff is 0s.
    # It contains the difference between the same samples and should not be counted to avoid
    # biasing the estimation.
    second_term = smpls_to_smpls_abs_diff.sum(dim=(-3, -2)) / (mc_samples * (mc_samples - 1))
    
    score = first_term - 0.5 * second_term 

    if aggregation == "none":
        return score
    if aggregation == "variable":
        return score.sum(dim=(0,1)) / mask.sum(dim=(0,1))
    elif aggregation == "all":
        return score.sum() / mask.sum()
    elif aggregation == "station":
        return score.sum(dim=0) / mask.sum(dim=0).clamp(min=1e-8)  # [station, feature]

    raise ValueError(f"Invalid aggregation: {aggregation}")



# TODO 
# Energy Score code, to be tested
# def energy_score(y_target, y_hat, mask=None):
    # mask_exp = mask.unsqueeze(-2)
            
    # # First term E(||X-y||)
    # smpls_to_tgt_dist = (y_hat - y_target.unsqueeze(-2)) * mask_exp
    # eucl_distance = torch.norm(smpls_to_tgt_dist, dim=-1)
    # first_term = eucl_distance.mean(dim=-1)

    # # Second term: E(||X-X'||)
    # xj = y_hat.unsqueeze(-2)
    # xi = y_hat.unsqueeze(-3)


    # smpls_diff = (xi - xj) * mask.unsqueeze(-2).unsqueeze(-2)
    # smpls_to_smpls_dist = torch.norm(smpls_diff, dim=-1)
    # # divide by n*(n-1) because the diagonal of smpls_to_smpls_dist is 0
    # second_term = smpls_to_smpls_dist.sum(dim=(-1,-2)) /  (mc_samples * (mc_samples -1))

    # # E(||X-y||) - 0.5 E(||X-X'||)

    # # TODO check this normalization
    # # valid is valid if at least one feature was preset
    # valid_locations = (mask.sum(dim=-1) > 0).float()
    # score = first_term - 0.5 * second_term
    # return (score * valid_locations).sum() / valid_locations.sum()


def variogram_weights(coords: torch.Tensor) -> torch.Tensor:
    """Pair weights w_ij = 1 / d_ij for the variogram score.

    Args:
        coords: [S, D] station coordinates.

    Returns:
        [S, S] weights (dtype of coords). The weight is 0 on the diagonal and
        for pairs with d_ij = 0.
    """
    # Float64 is necessary: float32 cdist gives 0.0 for LV95 stations 500 m apart.
    c = coords.double()
    d = torch.cdist(c, c)
    return torch.where(d > 0, 1.0 / d, torch.zeros_like(d)).to(coords.dtype)


def variogram_score(
    y_target: torch.Tensor,
    y_hat: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """Variogram score of order p for each batch element and variable.

    VS_p = sum_{i<j} w_ij * (|y_i - y_j|^p - mean_m |x_i,m - x_j,m|^p)^2,
    with w_ij normalized over the valid pairs. A pair is valid when both
    stations have mask 1. Lower is better.

    Reference: Scheuerer and Hamill (2015), Mon. Wea. Rev. 143, 1321-1334.

    Args:
        y_target: [B, S, V] observations. Masked entries can be NaN.
        y_hat: [B, S, M, V] ensemble members.
        mask: [B, S, V] float mask, 1 for a valid observation.
        weights: [S, S] pair weights from ``variogram_weights``.
        p: Order of the score.

    Returns:
        [B, V] scores. NaN where no valid pair has a nonzero weight.
    """
    S = y_hat.shape[1]
    upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=y_hat.device), 1)
    w_pair = (weights * upper)[..., None]  # [S, S, 1]
    valid = mask > 0.5
    return torch.stack(
        [
            _variogram_score_step(y_target[b], y_hat[b], valid[b], w_pair, p)
            for b in range(y_hat.shape[0])
        ]
    )


def fair_variogram_score(
    y_target: torch.Tensor,
    y_hat: torch.Tensor,
    mask: torch.Tensor,
    coords_km: torch.Tensor,
    p: float = 0.5,
    eps: float = 1e-3,
    dist_eps_km: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fair variogram score of order p for training.

    For each pair i < j, e_m = g(y_i - y_j) - g(x_i,m - x_j,m). The pair score
    is the mean of e_m * e_m' over m != m'. This estimator has no bias from
    the member count. The pair weights are 1 / (d_ij + dist_eps_km), with
    d_ij in km. The weights are normalized over the valid pairs. A pair is
    valid when both stations have mask 1.

    Args:
        y_target: [B, S, V] observations. Masked entries can be NaN.
        y_hat: [B, S, M, V] members. M must be at least 2.
        mask: [B, S, V] float mask, 1 for a valid observation.
        coords_km: [B, S, D] station coordinates in km.
        p: Order of the score.
        eps: Smoothing constant of the safe power g.
        dist_eps_km: Distance offset in km. It sets the largest weight.

    Returns:
        score: [B, V] scores. The value is 0 where has_pair is False.
        has_pair: [B, V] bool, True where at least one valid pair exists.
    """
    _, S, M, _ = y_hat.shape
    if M < 2:
        raise ValueError(f"Need at least 2 members, got {M}.")
    valid = mask > 0.5
    y = torch.where(valid, y_target, torch.zeros_like(y_target))

    def g(delta: torch.Tensor) -> torch.Tensor:
        # Smooth at delta = 0, because ST-ReLU gives exact zeros for dry members.
        out: torch.Tensor = (delta**2 + eps**2) ** (p / 2) - eps**p
        return out

    a = g(y[:, :, None] - y[:, None])  # [B, S, S, V]
    d = g(y_hat[:, :, None] - y_hat[:, None])  # [B, S, S, M, V]
    e = a[:, :, :, None] - d
    pair_score = (e.sum(dim=3) ** 2 - (e**2).sum(dim=3)) / (M * (M - 1))

    # Float64 is necessary for the distances. Float32 loses metres at LV95 size.
    c = coords_km.double()
    # The offset caps the weight of very close or coincident stations.
    omega = (1.0 / (torch.cdist(c, c) + dist_eps_km)).to(y_hat.dtype)  # [B, S, S]
    upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=y_hat.device), 1)
    pair = valid[:, :, None] & valid[:, None] & upper[None, :, :, None]
    w = torch.where(pair, omega[..., None], torch.zeros_like(pair_score))
    den = w.sum(dim=(1, 2))  # [B, V]
    num = torch.where(pair, w * pair_score, torch.zeros_like(pair_score)).sum(
        dim=(1, 2)
    )
    has_pair = den > 0
    score = torch.where(has_pair, num / den.clamp_min(1e-30), torch.zeros_like(num))
    return score, has_pair


def _variogram_score_step(
    y_target: torch.Tensor,
    y_hat: torch.Tensor,
    valid: torch.Tensor,
    w_pair: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Variogram score for one time step (see ``variogram_score``).

    Args:
        y_target: [S, V] observations. Entries with valid False can be NaN.
        y_hat: [S, M, V] ensemble members.
        valid: [S, V] bool, True for a valid observation.
        w_pair: [S, S, 1] pair weights, 0 on and below the diagonal.
        p: Order of the score.

    Returns:
        [V] scores. NaN where no valid pair has a nonzero weight.
    """
    M = y_hat.shape[1]
    y = torch.where(valid, y_target, torch.zeros_like(y_target))
    obs_v = (y[:, None] - y[None]).abs() ** p  # [S, S, V]
    # Loop over the members, so that the peak memory stays at [S, S, V].
    ens_v = torch.zeros_like(obs_v)
    for m in range(M):
        x = y_hat[:, m]  # [S, V]
        ens_v = ens_v + (x[:, None] - x[None]).abs() ** p
    ens_v = ens_v / M
    pair = valid[:, None] & valid[None]  # [S, S, V]
    w = torch.where(pair, w_pair, torch.zeros_like(obs_v))
    w_sum = w.sum(dim=(0, 1))  # [V]
    num = torch.where(w > 0, w * (obs_v - ens_v) ** 2, torch.zeros_like(obs_v)).sum(
        dim=(0, 1)
    )
    return torch.where(
        w_sum > 0, num / w_sum.clamp(min=1e-30), torch.full_like(num, float("nan"))
    )
