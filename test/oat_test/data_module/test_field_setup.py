from typing import Any

import torch

from oat.data_module.field import Field
from oat.data_module.field_setup import field_collate


def _sample(idx: int, mask: bool = True) -> dict[str, Any]:
    def mk(n: int) -> Field:
        c = torch.zeros(n, 3)
        v = torch.arange(float(n)).reshape(n, 1) + idx
        m = torch.ones(n, 1) if mask else None
        return Field(c, v, ["temperature"], m)

    return {
        "index": idx,
        "observation_data": {"fields": {"obs": mk(2)}},
        "query_data": {"fields": {"obs": mk(1)}},
    }


def test_field_collate_stacks_to_batched_fields() -> None:
    batch = field_collate([_sample(0), _sample(1)])
    obs = batch["observation_data"]["fields"]["obs"]
    assert obs.coords.shape == (2, 2, 3) and obs.values.shape == (2, 2, 1)
    assert obs.mask is not None and obs.mask.shape == (2, 2, 1)
    assert batch["query_data"]["fields"]["obs"].values.shape == (2, 1, 1)
    assert batch["index"].tolist() == [0, 1]


def test_field_collate_none_mask() -> None:
    batch = field_collate([_sample(0, mask=False), _sample(1, mask=False)])
    assert batch["observation_data"]["fields"]["obs"].mask is None
