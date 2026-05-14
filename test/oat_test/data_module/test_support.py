import pytest
import torch
from torch.distributions import constraints

from oat.data_module.support import VariableSupport


# --- VariableSupport ---


def test_multivariate_check_basic() -> None:
    """Each variable checked against its own bounds, results broadcast correctly."""
    lo = torch.tensor([0.0, -1.0, 2.0])
    hi = torch.tensor([1.0, 1.0, 5.0])
    s = VariableSupport(lo, hi)

    # shape [3] — one value per variable, along default dim=-1
    x = torch.tensor([0.5, 0.0, 3.0])
    assert s.check(x).all()

    x_bad = torch.tensor([1.5, 0.0, 3.0])   # first var out of bounds
    result = s.check(x_bad)
    assert torch.equal(result, torch.tensor([False, True, True]))


def test_multivariate_check_batched() -> None:
    """Works on [B, V] tensors with dim=-1 (default)."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    s = VariableSupport(lo, hi)

    x = torch.tensor([[0.5, 0.5], [1.5, 0.5]])   # shape [2, 2]
    result = s.check(x)
    expected = torch.tensor([[True, True], [False, True]])
    assert torch.equal(result, expected)


def test_multivariate_check_explicit_dim() -> None:
    """Variable dimension can be set to something other than -1."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    s = VariableSupport(lo, hi, dim=0)

    # shape [V, B] = [2, 3]
    x = torch.tensor([[0.5, 0.5, 1.5], [0.5, 0.5, 0.5]])
    result = s.check(x)
    expected = torch.tensor([[True, True, False], [True, True, True]])
    assert torch.equal(result, expected)


def test_multivariate_check_per_variable_open_flags() -> None:
    """left_open / right_open can be a bool tensor, one flag per variable."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    left_open = torch.tensor([True, False])
    right_open = torch.tensor([False, True])
    s = VariableSupport(lo, hi, left_open=left_open, right_open=right_open)

    # var 0: (0, 1]  →  0.0 is excluded
    # var 1: [0, 1)  →  1.0 is excluded
    x = torch.tensor([0.0, 1.0])
    assert torch.equal(s.check(x), torch.tensor([False, False]))

    x2 = torch.tensor([0.5, 0.5])
    assert s.check(x2).all()


def test_multivariate_check_scalar_open_flags() -> None:
    """Scalar bool open flags apply to all variables."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([1.0, 1.0])
    s = VariableSupport(lo, hi, left_open=True, right_open=True)

    x_boundary = torch.tensor([0.0, 1.0])
    assert not s.check(x_boundary).any()

    x_interior = torch.tensor([0.5, 0.5])
    assert s.check(x_interior).all()


def test_multivariate_constructor_validates_shapes() -> None:
    with pytest.raises(ValueError, match="1-D"):
        VariableSupport(torch.zeros(2, 2), torch.zeros(2))
    with pytest.raises(ValueError, match="same length"):
        VariableSupport(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="shape \\(2,\\)"):
        VariableSupport(
            torch.zeros(2),
            torch.ones(2),
            left_open=torch.tensor([True, False, True]),
        )
    with pytest.raises(ValueError, match="strictly less than"):
        VariableSupport(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))


def test_multivariate_infinite_bounds_are_open() -> None:
    """−inf lower bound and +inf upper bound are always treated as open."""
    from math import inf

    lo = torch.tensor([-inf, 0.0])
    hi = torch.tensor([0.0, inf])
    s = VariableSupport(lo, hi)

    # open flags must be forced on for the infinite sides
    assert s.left_open[0].item() is True
    assert s.right_open[1].item() is True

    # is_subset: (-inf, 0] ⊆ ℝ (real = (-inf, +inf) open) → True
    assert s.is_subset(constraints.real, idx=0) is True
    # is_subset: [0, +inf) ⊆ ℝ≥0 → True
    assert s.is_subset(constraints.nonnegative, idx=1) is True
