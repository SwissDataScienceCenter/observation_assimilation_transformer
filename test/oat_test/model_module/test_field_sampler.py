import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from oat.model_module.field_sampler import build_index


def test_build_index_returns_ckdtree_over_xy():
    coords = torch.tensor(
        [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]]
    )
    tree = build_index(coords)
    assert isinstance(tree, cKDTree)
    # tree indexes only (x, y): querying (0.9, 0.0) returns point 1
    _, idx = tree.query(np.array([[0.9, 0.0]]))
    assert idx[0] == 1


def test_build_index_uses_batch_element_zero():
    coords = torch.zeros(4, 3, 3)
    coords[0] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tree = build_index(coords)
    assert tree.n == 3


def test_build_index_rejects_unknown_distance():
    with pytest.raises(NotImplementedError, match="euclidean"):
        build_index(torch.zeros(3, 3), distance="haversine")
