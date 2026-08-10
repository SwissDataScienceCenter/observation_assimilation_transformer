"""Field-path data/transform setup + Field-aware collate."""

from __future__ import annotations

from typing import Any

import torch

from oat.data_module.field import Field

_GROUPS = ("observation_data", "query_data")


def field_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"index": torch.tensor([s["index"] for s in samples])}
    for group in _GROUPS:
        names = list(samples[0][group]["fields"].keys())
        fields: dict[str, Field] = {}
        for name in names:
            flist = [s[group]["fields"][name] for s in samples]
            coords = torch.stack([f.coords for f in flist], dim=0)
            values = torch.stack([f.values for f in flist], dim=0)
            mask = (
                torch.stack([f.mask for f in flist], dim=0)
                if flist[0].mask is not None
                else None
            )
            fields[name] = Field(coords, values, list(flist[0].var_names), mask)
        out[group] = {"fields": fields}
    return out
