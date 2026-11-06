from oat.predictor_module.functional_metrics import (
    METRICS,
    fair_variogram_score,
    variogram_score,
    variogram_weights,
)

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


def test_variogram_weights_lv95_precision():
    # float32 torch.cdist gives 0.0 for this pair (checked 2026-11-04).
    # float64 must give 500 m.
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


def _g(d, p=0.5, eps=1e-3):
    return (d**2 + eps**2) ** (p / 2) - eps**p


def _fair_ref(y, x, coords, valid=None, dist_eps=1.0):
    """Loop reference for one sample and one variable. y [S], x [S, M]."""
    S, M = x.shape
    num, den = 0.0, 0.0
    for i in range(S):
        for j in range(i + 1, S):
            if valid is not None and not (valid[i] and valid[j]):
                continue
            dist = float(torch.dist(coords[i].double(), coords[j].double()))
            w = 1.0 / (dist + dist_eps)
            a = _g(y[i] - y[j])
            e = [a - _g(x[i, m] - x[j, m]) for m in range(M)]
            vs = sum(e[m] * e[k] for m in range(M) for k in range(M) if m != k)
            num += w * float(vs) / (M * (M - 1))
            den += w
    return num / den


def _corr_samples(pos, n, scale=2.0):
    d = torch.cdist(pos, pos)
    K = torch.exp(-d / scale) + 1e-6 * torch.eye(len(pos))
    L = torch.linalg.cholesky(K)
    return torch.randn(n, len(pos)) @ L.T


def _line(S):
    pos = torch.arange(S, dtype=torch.float32)[:, None]
    return pos, torch.cat([pos, torch.zeros_like(pos)], dim=1)


def test_fair_vs_matches_member_pair_loop():
    torch.manual_seed(0)
    y = torch.randn(1, 3, 1)
    x = torch.randn(1, 3, 3, 1)
    coords = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]]])
    score, has_pair = fair_variogram_score(y, x, torch.ones(1, 3, 1), coords)
    expected = _fair_ref(y[0, :, 0], x[0, :, :, 0], coords[0])
    assert torch.allclose(score, torch.tensor([[expected]]), atol=1e-6)
    assert has_pair.all()


def test_fair_vs_unbiased_in_members():
    torch.manual_seed(1)
    pos, coords = _line(4)
    n = 4000
    y = _corr_samples(pos, n)[..., None]
    c = coords[None].expand(n, -1, -1)
    mask = torch.ones(n, 4, 1)

    def members(M):
        return torch.stack([_corr_samples(pos, n) for _ in range(M)], dim=2)[..., None]

    f2, _ = fair_variogram_score(y, members(2), mask, c)
    f64, _ = fair_variogram_score(y, members(64), mask, c)
    se = (f2.std() ** 2 / n + f64.std() ** 2 / n) ** 0.5
    assert abs(f2.mean() - f64.mean()) < 4 * se
    x2 = members(2)
    pw = (_g(y[:, :, None] - y[:, None]) - _g(x2[:, :, None] - x2[:, None]).mean(3))
    assert pw.shape[0] == n  # plain form built from the same members
    S = 4
    up = torch.triu(torch.ones(S, S, dtype=torch.bool), 1)
    om = 1.0 / (torch.cdist(coords, coords) + 1.0)
    w = (om * up)[None, :, :, None]
    plain = ((w * pw**2).sum((1, 2)) / w.sum()).mean()
    assert plain > f64.mean()


def test_fair_vs_prefers_correct_correlation():
    torch.manual_seed(2)
    S, n, M = 30, 500, 16
    pos, coords = _line(S)
    c = coords[None].expand(n, -1, -1)
    mask = torch.ones(n, S, 1)
    y = _corr_samples(pos, n)[..., None]
    correct = torch.stack([_corr_samples(pos, n) for _ in range(M)], dim=2)[..., None]
    indep = torch.randn(n, S, M, 1)
    full = torch.randn(n, 1, M, 1).expand(n, S, M, 1)
    sc, _ = fair_variogram_score(y, correct, mask, c)
    si, _ = fair_variogram_score(y, indep, mask, c)
    sf, _ = fair_variogram_score(y, full, mask, c)
    assert sc.mean() < si.mean()
    assert sc.mean() < sf.mean()


def test_fair_vs_finite_gradient_for_equal_and_dry_members():
    x = torch.zeros(2, 4, 3, 1, requires_grad=True)
    coords = torch.rand(2, 4, 2) * 10
    score, _ = fair_variogram_score(
        torch.zeros(2, 4, 1), x, torch.ones(2, 4, 1), coords
    )
    assert torch.allclose(score, torch.zeros_like(score))
    score.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_fair_vs_coincident_stations():
    torch.manual_seed(3)
    y = torch.randn(1, 3, 1)
    x = torch.randn(1, 3, 4, 1, requires_grad=True)
    coords = torch.tensor([[[5.0, 5.0], [5.0, 5.0], [9.0, 5.0]]])
    score, _ = fair_variogram_score(y, x, torch.ones(1, 3, 1), coords)
    expected = _fair_ref(y[0, :, 0], x[0, :, :, 0].detach(), coords[0])
    assert torch.isfinite(score).all()
    assert torch.allclose(score, torch.tensor([[expected]]), atol=1e-6)
    score.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_fair_vs_masked_nan_station_equals_score_without_it():
    torch.manual_seed(4)
    y = torch.randn(1, 3, 1)
    x = torch.randn(1, 3, 4, 1, requires_grad=True)
    coords = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [7.0, 1.0]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    y_nan = y.clone()
    y_nan[0, 1, 0] = float("nan")
    score, _ = fair_variogram_score(y_nan, x, mask, coords)
    keep = [0, 2]
    ref, _ = fair_variogram_score(
        y[:, keep], x[:, keep], torch.ones(1, 2, 1), coords[:, keep]
    )
    assert torch.allclose(score, ref, atol=1e-6)
    score.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_fair_vs_no_valid_pair():
    torch.manual_seed(5)
    y = torch.randn(2, 3, 2)
    x = torch.randn(2, 3, 3, 2, requires_grad=True)
    coords = torch.rand(2, 3, 2) * 20
    mask = torch.ones(2, 3, 2)
    mask[0, 1:, 0] = 0.0
    y[0, 1:, 0] = float("nan")
    score, has_pair = fair_variogram_score(y, x, mask, coords)
    assert not has_pair[0, 0]
    assert score[0, 0] == 0
    assert has_pair.sum() == 3
    assert torch.isfinite(score).all()
    score.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[0, :, :, 0] == 0).all()


def test_fair_vs_lv95_precision():
    torch.manual_seed(6)
    S, M = 30, 4
    # Above 25 rows, float32 cdist uses the matmul path and loses metres.
    coords_m = torch.rand(1, S, 2) * 1e5 + torch.tensor([2.6e6, 1.2e6])
    coords_m[0, 1] = coords_m[0, 0] + torch.tensor([500.0, 0.0])
    coords_km = coords_m * 0.001
    bad = torch.cdist(coords_km[0], coords_km[0])[0, 1]
    assert abs(bad.item() - 0.5) > 0.1  # float32 cdist is wrong for this pair
    y = torch.randn(1, S, 1)
    x = torch.randn(1, S, M, 1)
    mask = torch.ones(1, S, 1)
    s32, _ = fair_variogram_score(y, x, mask, coords_km)
    s64, _ = fair_variogram_score(
        y.double(), x.double(), mask.double(), coords_km.double()
    )
    assert torch.allclose(s32.double(), s64, atol=1e-6)
