import math

import pytest
import torch

from oat.model_module.output_head import OutputHead
from oat.model_module.query_noise import RFFNoiseField, SampleHead
from oat.predictor_module import functional_metrics
from oat.predictor_module.functional_metrics import (
    fair_variogram_score,
    mc_crps,
)


def _field(scales=(10.0,), km_per_unit=1.0, cps=4, K=128):
    return RFFNoiseField(scales, km_per_unit, cps, K)


@pytest.mark.parametrize("D", [2, 3])
def test_draw_and_evaluate_shapes(D):
    f = _field(scales=(2.0, 8.0), cps=3, K=16)
    d = f.draw(2, 5, D, torch.device("cpu"), torch.float32)
    assert d.omega.shape == (2, 5, 6, 16, D)
    assert d.evaluate(torch.randn(2, 7, D)).shape == (2, 7, 5, 6)


def test_unit_variance():
    g = torch.Generator().manual_seed(0)
    d = _field().draw(1, 4000, 2, torch.device("cpu"), torch.float32, g)
    eps = d.evaluate(torch.zeros(1, 1, 2) + 3.0)
    assert torch.allclose(eps.var(dim=2), torch.ones(1, 1, 4), atol=0.1)


def test_correlation_matches_gaussian_kernel():
    g = torch.Generator().manual_seed(0)
    d = _field(scales=(10.0,), cps=1, K=256).draw(
        1, 8000, 2, torch.device("cpu"), torch.float32, g
    )
    x = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]])
    eps = d.evaluate(x)[0, :, :, 0]
    rho = torch.corrcoef(eps)[0, 1]
    assert abs(rho - math.exp(-0.5)) < 0.05


def test_same_draw_same_values_new_draw_differs():
    f = _field()
    d = f.draw(1, 3, 2, torch.device("cpu"), torch.float32)
    x = torch.randn(1, 6, 2)
    assert torch.equal(d.evaluate(x), d.evaluate(x))
    d2 = f.draw(1, 3, 2, torch.device("cpu"), torch.float32)
    assert not torch.allclose(d.evaluate(x), d2.evaluate(x))


def test_km_per_unit_scaling():
    x_km = torch.rand(1, 9, 2) * 50
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    e_m = (
        _field(km_per_unit=0.001)
        .draw(1, 3, 2, torch.device("cpu"), torch.float64, g1)
        .evaluate(x_km.double() * 1000)
    )
    e_km = (
        _field(km_per_unit=1.0)
        .draw(1, 3, 2, torch.device("cpu"), torch.float64, g2)
        .evaluate(x_km.double())
    )
    assert torch.allclose(e_m, e_km, atol=1e-8)


@pytest.mark.parametrize("bad", ["batch", "dim", "broadcast_batch"])
def test_evaluate_rejects_mismatched_coords(bad):
    batch = 1 if bad == "broadcast_batch" else 2
    d = _field().draw(batch, 3, 2, torch.device("cpu"), torch.float32)
    coords = torch.randn(3, 4, 2)
    if bad == "dim":
        coords = torch.randn(2, 4, 3)
    with pytest.raises(ValueError):
        d.evaluate(coords)


def _head(dists=("normal", "zero_inflated_lognormal"), **kw):
    v2d = {f"v{i}": d for i, d in enumerate(dists)}
    return SampleHead(
        in_features=8, noise_channels=4, output_head=OutputHead(8, v2d), **kw
    )


def test_sample_head_loss_matches_mc_crps():
    torch.manual_seed(0)
    head = _head(("normal",))
    y = head(torch.randn(2, 4, 8), torch.randn(2, 4, 5, 4))
    target = torch.randn(2, 4, 1)
    _, loss, _ = head.loss(y, target, torch.ones(2, 4, 1))
    assert torch.allclose(loss, mc_crps(target, y, aggregation="all"))


def test_sample_head_shape_and_exact_zeros():
    torch.manual_seed(0)
    y = _head()(torch.randn(2, 5, 8), torch.randn(2, 5, 6, 4))
    assert y.shape == (2, 5, 6, 2)
    assert (y[..., 1] >= 0).all() and (y[..., 1] == 0).any()
    assert (y[..., 0] < 0).any()


def test_straight_through_relu_gives_gradient_for_dry_member():
    head = _head(("zero_inflated_lognormal",))
    h = torch.randn(1, 1, 8, requires_grad=True)
    with torch.no_grad():
        head.mlp[-1].bias.fill_(-100.0)
    y = head(h, torch.randn(1, 1, 4, 4))
    _, loss, _ = head.loss(y, torch.full((1, 1, 1), 2.0), torch.ones(1, 1, 1))
    loss.backward()
    assert h.grad.abs().sum() > 0


def test_masked_nan_target_gives_finite_loss():
    head = _head(("normal",))
    y = head(torch.randn(1, 3, 8), torch.randn(1, 3, 4, 4))
    target = torch.tensor([[[1.0], [float("nan")], [0.5]]])
    mask = torch.tensor([[[1.0], [0.0], [1.0]]])
    per_var, loss, _ = head.loss(y, target, mask)
    assert torch.isfinite(loss) and set(per_var) == {"v0"}
    loss.backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_sample_head_does_not_store_output_head():
    assert not any("output_head" in k for k in _head().state_dict())


def test_sample_head_vs_term_matches_formula():
    torch.manual_seed(0)
    head = _head(vs_weight=0.3)
    members = head(torch.randn(2, 4, 8), torch.randn(2, 4, 5, 4))
    target = torch.randn(2, 4, 2).abs()
    mask = torch.ones(2, 4, 2)
    mask[0, 1, 0] = 0.0
    coords = torch.rand(2, 4, 2) * 50.0
    per_var, total, vs = head.loss(members, target, mask, coords)
    safe = torch.where(mask > 0.5, target, torch.zeros_like(target))
    crps = mc_crps(safe, members, mask=None, aggregation="none")
    crps = torch.where(mask > 0.5, crps, torch.zeros_like(crps))
    score, has_pair = fair_variogram_score(target, members, mask, coords, p=0.5)
    expected = torch.zeros(())
    for i, name in enumerate(head.variable_names):
        l_crps = OutputHead.loss_normalization_per_sample(
            crps[..., i : i + 1], mask[..., i : i + 1]
        )
        hp = has_pair[:, i]
        l_vs = score[:, i][hp].mean() if hp.any() else torch.zeros(())
        assert torch.allclose(vs[name], l_vs.detach())
        s_v = head.loss_scaling[i]
        expected = expected + s_v * (l_crps + 0.3 * l_vs)
    assert set(vs) == set(head.variable_names)
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(sum(per_var.values()), total.detach(), atol=1e-6)


def test_sample_head_vs_off_does_not_call_function(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("VS must not run")

    monkeypatch.setattr(functional_metrics, "fair_variogram_score", boom)
    head = _head(("normal",))
    y = head(torch.randn(1, 3, 8), torch.randn(1, 3, 4, 4))
    _, _, vs = head.loss(y, torch.randn(1, 3, 1), torch.ones(1, 3, 1))
    assert vs == {}


def test_sample_head_vs_needs_coords():
    head = _head(("normal",), vs_weight=0.1)
    y = head(torch.randn(1, 3, 8), torch.randn(1, 3, 4, 4))
    with pytest.raises(ValueError):
        head.loss(y, torch.randn(1, 3, 1), torch.ones(1, 3, 1), None)


def test_sample_head_vs_dry_members_finite_gradients():
    head = _head(("zero_inflated_lognormal",), vs_weight=1.0)
    with torch.no_grad():
        head.mlp[-1].bias.fill_(-100.0)
    h = torch.randn(1, 4, 8, requires_grad=True)
    y = head(h, torch.randn(1, 4, 4, 4))
    assert (y == 0).all()
    _, loss, _ = head.loss(
        y, torch.zeros(1, 4, 1), torch.ones(1, 4, 1), torch.rand(1, 4, 2) * 20
    )
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert torch.isfinite(h.grad).all()


def _mix_head(S=2, C=4, nugget_index=0, **kw):
    v2d = {"v0": "normal"}
    return SampleHead(
        in_features=8,
        noise_channels=S * C,
        output_head=OutputHead(8, v2d),
        mixing="scale_mixture",
        num_scales=S,
        nugget_index=nugget_index,
        **kw,
    )


def test_mixture_init_pi_on_nugget():
    head = _mix_head(S=5, nugget_index=0)
    pi = head.pi
    assert torch.allclose(pi.sum(), torch.tensor(1.0))
    assert abs(pi[0].item() - 1 / (1 + 4 * math.exp(-4.0))) < 1e-6
    assert torch.allclose(pi[1:], pi[1].expand(4))


def test_mixture_output_has_C_channels_and_unit_variance():
    torch.manual_seed(0)
    head = _mix_head(S=3, C=4)
    with torch.no_grad():
        head.mixture_delta.copy_(torch.tensor([0.0, 3.0, 4.0]))
    out = head.mix_noise(torch.randn(1, 2000, 8, 12))
    assert out.shape == (1, 2000, 8, 4)
    assert abs(out.var().item() - 1.0) < 0.05


def test_mixture_correlation_matches_mixture_kernel():
    # The field and the head together: the head must group the channels of
    # one length scale. Check each output channel, because a wrong grouping
    # can give the correct mean over the channels.
    torch.manual_seed(0)
    field = RFFNoiseField((0.1, 20.0), 1.0, 4, 512)
    head = _mix_head(S=2, C=4)
    with torch.no_grad():
        head.mixture_delta.copy_(torch.tensor([0.0, 4.0]))  # pi = [0.5, 0.5]
    coords = torch.tensor([[[0.0, 0.0], [10.0, 0.0]]]).expand(4000, 2, 2).contiguous()
    draw = field.draw(4000, 1, 2, torch.device("cpu"), torch.float32)
    mixed = head.mix_noise(draw.evaluate(coords))  # [4000, 2, 1, 4]
    expected = 0.5 * math.exp(-(10.0**2) / (2 * 20.0**2))
    for c in range(4):
        a, b = mixed[:, 0, 0, c], mixed[:, 1, 0, c]
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        assert abs(corr - expected) < 0.08, (c, corr)


def test_mixture_needs_matching_channels():
    with pytest.raises(ValueError):
        SampleHead(
            in_features=8,
            noise_channels=10,
            output_head=OutputHead(8, {"v0": "normal"}),
            mixing="scale_mixture",
            num_scales=3,
        )


def test_channels_mode_has_no_new_state():
    head = _head()
    assert head.pi is None
    assert not any("mixture" in k for k in head.state_dict())
    eps = torch.randn(2, 3, 5, 4)
    assert torch.equal(head.mix_noise(eps), eps)


def test_mixture_delta_gets_gradient():
    torch.manual_seed(0)
    head = _mix_head(S=2, C=4)
    y = head(torch.randn(2, 4, 8), torch.randn(2, 4, 5, 8))
    y.sum().backward()
    assert head.mixture_delta.grad is not None


@pytest.mark.parametrize(
    "epoch,expected",
    [(0, 0.0), (9, 0.0), (10, 0.05), (14, 0.25), (19, 0.5), (30, 0.5)],
)
def test_train_vs_weight_schedule(epoch, expected):
    head = _head(("normal",), vs_weight=0.5, vs_warmup_epochs=10, vs_ramp_epochs=10)
    head.epoch = float(epoch)
    assert head.train_vs_weight() == pytest.approx(expected)


def test_no_ramp_jumps_to_full_weight():
    head = _head(("normal",), vs_weight=0.5, vs_warmup_epochs=3)
    head.epoch = 3.0
    assert head.train_vs_weight() == 0.5


def test_warmup_train_has_no_vs_eval_has_vs():
    torch.manual_seed(0)
    head = _head(("normal",), vs_weight=0.5, vs_warmup_epochs=10)
    y = head(torch.randn(2, 4, 8), torch.randn(2, 4, 5, 4))
    t, m, xy = torch.randn(2, 4, 1), torch.ones(2, 4, 1), torch.randn(2, 4, 2)
    head.train()
    _, _, vs_train = head.loss(y, t, m, xy)
    head.eval()
    _, _, vs_eval = head.loss(y, t, m, xy)
    assert vs_train == {} and set(vs_eval) == {"v0"}


def test_epoch_is_not_state():
    head = _head(vs_warmup_epochs=2)
    head.epoch = 5.0
    assert "epoch" not in head.state_dict()


@pytest.mark.parametrize("kw", [{"vs_warmup_epochs": -1}, {"vs_ramp_epochs": -1}])
def test_negative_schedule_rejected(kw):
    with pytest.raises(ValueError):
        _head(**kw)
