import torch

from oat.model_module.model import _search_coords


def test_search_coords_prefers_search_coords():
    g = {
        "coords": torch.zeros(1, 4, 2),
        "search_coords": torch.ones(1, 4, 3),
    }
    out = _search_coords(g)
    assert out.shape == (1, 4, 3)
    assert torch.equal(out, g["search_coords"])


def test_search_coords_falls_back_to_coords():
    g = {"coords": torch.zeros(1, 4, 2)}
    out = _search_coords(g)
    assert out.shape == (1, 4, 2)
    assert torch.equal(out, g["coords"])
