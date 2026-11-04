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


from oat.predictor_module.functional_metrics import (  # noqa: E402
    variogram_score,
    variogram_weights,
)


def test_variogram_weights_lv95_precision():
    # float32 torch.cdist gives 0.0 for this pair (checked 2026-11-04); float64 must give 500 m
    c = torch.tensor(
        [[2.6e6, 1.2e6], [2.6e6 + 500.0, 1.2e6], [2.7e6, 1.3e6]], dtype=torch.float32
    )
    w = variogram_weights(c)
    assert abs(1.0 / w[0, 1].item() - 500.0) < 1.0


def _vs_coords():
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])


def _vs_case():
    return variogram_weights(_vs_coords())


def test_variogram_weights_zero_diagonal_and_colocated():
    w = variogram_weights(torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]]))
    assert torch.all(torch.diagonal(w) == 0)
    assert w[0, 1] == 0
    assert w[0, 2].item() == pytest.approx(0.2)


def test_variogram_score_zero_when_members_equal_target():
    torch.manual_seed(0)
    y = torch.randn(2, 3, 1)
    y_hat = y.unsqueeze(-2).expand(2, 3, 4, 1)
    vs = variogram_score(y, y_hat, torch.ones(2, 3, 1), _vs_case())
    assert torch.allclose(vs, torch.zeros(2, 1), atol=1e-6)


def test_variogram_score_matches_hand_computation():
    torch.manual_seed(1)
    p = 0.5
    y = torch.randn(1, 3, 1)
    y_hat = torch.randn(1, 3, 2, 1)
    w = _vs_case()
    pairs = [(0, 1), (0, 2), (1, 2)]
    w_sum = sum(w[i, j].item() for i, j in pairs)
    expected = 0.0
    for i, j in pairs:
        obs = abs(y[0, i, 0].item() - y[0, j, 0].item()) ** p
        ens = sum(
            abs(y_hat[0, i, m, 0].item() - y_hat[0, j, m, 0].item()) ** p
            for m in range(2)
        ) / 2
        expected += w[i, j].item() / w_sum * (obs - ens) ** 2
    vs = variogram_score(y, y_hat, torch.ones(1, 3, 1), w)
    assert vs[0, 0].item() == pytest.approx(expected, rel=1e-5)


def test_variogram_score_masked_nan_station_equals_score_without_it():
    torch.manual_seed(2)
    coords = _vs_coords()
    y = torch.randn(1, 3, 1)
    y_hat = torch.randn(1, 3, 4, 1)
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    y_nan = y.clone()
    y_nan[0, 1, 0] = float("nan")
    vs = variogram_score(y_nan, y_hat, mask, variogram_weights(coords))
    keep = [0, 2]
    ref = variogram_score(
        y[:, keep],
        y_hat[:, keep],
        torch.ones(1, 2, 1),
        variogram_weights(coords[keep]),
    )
    assert torch.isfinite(vs).all()
    assert torch.allclose(vs, ref)


def test_variogram_score_invariant_to_coordinate_scale():
    torch.manual_seed(3)
    coords = _vs_coords()
    y = torch.randn(2, 3, 2)
    y_hat = torch.randn(2, 3, 5, 2)
    mask = torch.ones(2, 3, 2)
    a = variogram_score(y, y_hat, mask, variogram_weights(coords))
    b = variogram_score(y, y_hat, mask, variogram_weights(coords * 1000.0))
    assert torch.allclose(a, b, rtol=1e-4)


def test_variogram_score_nan_with_fewer_than_two_valid_stations():
    torch.manual_seed(4)
    y = torch.randn(2, 3, 2)
    y_hat = torch.randn(2, 3, 4, 2)
    mask = torch.ones(2, 3, 2)
    mask[0, 1:, 0] = 0
    vs = variogram_score(y, y_hat, mask, _vs_case())
    assert torch.isnan(vs[0, 0])
    assert torch.isfinite(vs.flatten()[1:]).all()


def test_variogram_score_prefers_correct_dependence():
    torch.manual_seed(5)
    S, T, M = 30, 200, 20
    pos = torch.arange(S, dtype=torch.float32)[:, None]
    coords = torch.cat([pos, torch.zeros_like(pos)], dim=1)
    d = torch.cdist(pos, pos)
    K = torch.exp(-(d**2) / (2 * 5.0**2)) + 1e-6 * torch.eye(S)
    L = torch.linalg.cholesky(K)
    truth = (torch.randn(T, S) @ L.T)[..., None]  # [T, S, 1]
    ens_a = torch.einsum("ij,tmj->tmi", L, torch.randn(T, M, S)).permute(0, 2, 1)
    ens_a = ens_a[..., None]  # [T, S, M, 1]
    ens_b = torch.randn(T, S, M, 1)
    mask = torch.ones(T, S, 1)
    w = variogram_weights(coords)
    vs_a = variogram_score(truth, ens_a, mask, w)
    vs_b = variogram_score(truth, ens_b, mask, w)
    assert vs_a.mean() < vs_b.mean()
