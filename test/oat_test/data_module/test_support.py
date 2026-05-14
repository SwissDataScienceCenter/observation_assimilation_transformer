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
    s = VariableSupport(lo, hi, lower_open=left_open, upper_open=right_open)

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
    s = VariableSupport(lo, hi, lower_open=True, upper_open=True)

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
            lower_open=torch.tensor([True, False, True]),
        )
    with pytest.raises(ValueError, match="strictly less than"):
        VariableSupport(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))


def test_multivariate_infinite_bounds_are_open() -> None:
    """-inf lower bound and +inf upper bound are always treated as open."""
    from math import inf

    lo = torch.tensor([-inf, 0.0])
    hi = torch.tensor([0.0, inf])
    s = VariableSupport(lo, hi)

    # open flags must be forced on for the infinite sides
    assert s.lower_open[0].item() is True
    assert s.upper_open[1].item() is True

    # is_subset: (-inf, 0] ⊆ R (real = (-inf, +inf) open) → True
    assert s.is_subset(constraints.real, idx=0) is True
    # is_subset: [0, +inf) ⊆ R≥0 → True
    assert s.is_subset(constraints.nonnegative, idx=1) is True


def test_multivariate_is_subset_closed_bounds() -> None:
    """Closed bounds: endpoint must satisfy target's bound+openness."""
    lo = torch.tensor([0.0, 0.2])
    hi = torch.tensor([1.0, 0.8])
    s = VariableSupport(lo, hi)  # both closed

    # [0, 1] ⊆ [0, 1]  → True  (lo == target.lo, closed both sides)
    assert s.is_subset(constraints.interval(0.0, 1.0), idx=0) is True
    # [0.2, 0.8] ⊆ [0, 1]  → True
    assert s.is_subset(constraints.interval(0.0, 1.0), idx=1) is True
    # [0, 1] ⊆ [0, 1)  → False  (hi == target.hi but self is closed, target is open)
    assert s.is_subset(constraints.half_open_interval(0.0, 1.0), idx=0) is False
    # [0, 1] ⊆ (0, +∞)  → False  (lo == target.lo=0 but self is closed, target is open)
    assert s.is_subset(constraints.positive, idx=0) is False
    # [0, 1] ⊆ [0, +∞)  → True
    assert s.is_subset(constraints.nonnegative, idx=0) is True


def test_multivariate_is_subset_open_bounds() -> None:
    """Open bounds: openness of self relaxes the equality condition."""
    lo = torch.tensor([0.0])
    hi = torch.tensor([1.0])
    s_lo_open = VariableSupport(lo, hi, lower_open=True)
    s_hi_open = VariableSupport(lo, hi, upper_open=True)
    s_both_open = VariableSupport(lo, hi, lower_open=True, upper_open=True)

    # (0, 1] ⊆ (0, +∞)  → True  (lo == target.lo=0, self open satisfies condition)
    assert s_lo_open.is_subset(constraints.positive, idx=0) is True
    # (0, 1] ⊆ [0.5, +∞)  → False  (lo=0 < target.lo=0.5)
    assert s_lo_open.is_subset(constraints.greater_than_eq(0.5), idx=0) is False

    # [0, 1) ⊆ [0, 1]  → True  (hi == target.hi=1, self open satisfies condition)
    assert s_hi_open.is_subset(constraints.interval(0.0, 1.0), idx=0) is True
    # [0, 1) ⊆ [0, 0.5]  → False  (hi=1 > target.hi=0.5)
    assert s_hi_open.is_subset(constraints.interval(0.0, 0.5), idx=0) is False

    # (0, 1) ⊆ (0, +∞)  → True
    assert s_both_open.is_subset(constraints.positive, idx=0) is True
    # (0, 1) ⊆ R  → True
    assert s_both_open.is_subset(constraints.real, idx=0) is True


def test_multivariate_is_subset_discrete_raises() -> None:
    lo = torch.tensor([0.0])
    hi = torch.tensor([1.0])
    s = VariableSupport(lo, hi)
    with pytest.raises(ValueError, match="continuous constraints"):
        s.is_subset(constraints.boolean, idx=0)


def test_multivariate_is_subset_no_idx_scalar_constraint_checks_all_dimensions() -> None:
    lo = torch.tensor([0.0, 0.2])
    hi = torch.tensor([1.0, 0.8])
    s = VariableSupport(lo, hi)

    assert s.is_subset(constraints.interval(0.0, 1.0)) is True
    assert s.is_subset(constraints.half_open_interval(0.0, 1.0)) is False


def test_multivariate_is_subset_no_idx_variable_support_single_dimension_checks_all(
) -> None:
    base = VariableSupport(torch.tensor([0.0, 0.2]), torch.tensor([1.0, 0.8]))
    one_dim = VariableSupport(torch.tensor([0.0]), torch.tensor([1.0]))

    assert base.is_subset(one_dim) is True


def test_multivariate_is_subset_no_idx_variable_support_equal_dimension_componentwise(
) -> None:
    base = VariableSupport(torch.tensor([0.0, 0.2]), torch.tensor([1.0, 0.8]))
    wider = VariableSupport(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
    tighter = VariableSupport(torch.tensor([0.0, 0.3]), torch.tensor([1.0, 0.7]))

    assert base.is_subset(wider) is True
    assert base.is_subset(tighter) is False


def test_multivariate_is_subset_no_idx_variable_support_dimension_mismatch_raises(
) -> None:
    base = VariableSupport(torch.tensor([0.0, 0.2]), torch.tensor([1.0, 0.8]))
    mismatch = VariableSupport(
        torch.tensor([0.0, 0.1, 0.2]),
        torch.tensor([1.0, 0.9, 0.8]),
    )

    with pytest.raises(ValueError, match="equal dimension"):
        base.is_subset(mismatch)


def test_multivariate_is_subset_no_idx_discrete_constraint_raises() -> None:
    s = VariableSupport(torch.tensor([0.0, 0.2]), torch.tensor([1.0, 0.8]))

    with pytest.raises(ValueError, match="continuous constraints"):
        s.is_subset(constraints.boolean)
