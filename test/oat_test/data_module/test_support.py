import pytest
import torch
from torch.distributions import constraints

from oat.data_module.support import IntervalSupport, create_variable_support


def test_default_interval_is_unbounded() -> None:
    c = IntervalSupport()
    x = torch.tensor([-1e6, 0.0, 1e6])
    assert c.check(x).all()


def test_open_closed_bound_checks() -> None:
    closed = IntervalSupport(lower_bound=0.0, upper_bound=1.0)
    half_open = IntervalSupport(lower_bound=0.0, upper_bound=1.0, right_open=True)
    fully_open = IntervalSupport(
        lower_bound=0.0,
        upper_bound=1.0,
        left_open=True,
        right_open=True,
    )

    x = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(closed.check(x), torch.tensor([True, True, True]))
    assert torch.equal(half_open.check(x), torch.tensor([True, True, False]))
    assert torch.equal(fully_open.check(x), torch.tensor([False, True, False]))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (IntervalSupport(0.0, 1.0), IntervalSupport(0.0, 1.0), True),
        (IntervalSupport(0.0, 1.0, right_open=True), IntervalSupport(0.0, 1.0), True),
        (IntervalSupport(0.0, 1.0), IntervalSupport(0.0, 1.0, right_open=True), False),
        (IntervalSupport(0.2, 0.8), IntervalSupport(0.0, 1.0), True),
        (IntervalSupport(0.2, 1.1), IntervalSupport(0.0, 1.0), False),
        (IntervalSupport(lower_bound=0.0), IntervalSupport(lower_bound=-1.0), True),
    ],
)
def test_is_subset_rules(a: IntervalSupport, b: IntervalSupport, expected: bool) -> None:
    assert a.is_subset(b) is expected


def test_factory_from_string_key() -> None:
    c = create_variable_support("interval", lower_bound=0.0, upper_bound=1.0)
    assert isinstance(c, IntervalSupport)


def test_factory_from_torch_constraint() -> None:
    c = create_variable_support(constraints.half_open_interval(0.0, 1.0))
    assert isinstance(c, IntervalSupport)
    assert c.left_open is False
    assert c.right_open is True


def test_discrete_constraints_raise_in_factory_and_subset() -> None:
    with pytest.raises(ValueError, match="continuous constraints"):
        create_variable_support(constraints.boolean)

    with pytest.raises(ValueError, match="continuous constraints"):
        IntervalSupport().is_subset(constraints.boolean)
