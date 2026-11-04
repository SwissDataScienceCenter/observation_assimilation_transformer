import math

import pytest
import torch

from oat.model_module.query_noise import RFFNoiseField


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


@pytest.mark.parametrize("bad", ["batch", "dim"])
def test_evaluate_rejects_mismatched_coords(bad):
    d = _field().draw(2, 3, 2, torch.device("cpu"), torch.float32)
    coords = torch.randn(3, 4, 2) if bad == "batch" else torch.randn(2, 4, 3)
    with pytest.raises(RuntimeError):
        d.evaluate(coords)
