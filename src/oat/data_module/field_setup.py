"""Field-path data/transform setup + Field-aware collate."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from oat.data_module import transform as T
from oat.data_module.field import Field
from oat.data_module.field_data_split import create_field_splits

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


def build_two_group_transform(
    inner_by_field: dict[str, T.Transform],
) -> T.NestedTransform:
    """Wrap per-field inner transforms as independent obs/query FieldValueTransforms.

    Each group (``observation_data``, ``query_data``) gets its own
    ``FieldValueTransform`` per field, built from a deep copy of the shared
    inner transform, so the same ``nn.Module`` is never registered under two
    ``ModuleDict`` paths.
    """

    def wrap() -> dict[str, T.Transform]:
        return {
            n: T.FieldValueTransform(copy.deepcopy(inner))
            for n, inner in inner_by_field.items()
        }

    return T.NestedTransform(
        {
            "observation_data": {"fields": wrap()},
            "query_data": {"fields": wrap()},
        }
    )


def validate_field_config_against_scalars(dataset: Any, cfg: DictConfig) -> None:
    """Check that field-config channel order matches the legacy scalar var lists.

    Requires ``roles`` to define ``target`` and ``global``; ``geo_bias`` is
    only checked when present, so field configs without a geo field are
    tolerated.
    """
    fc = dataset.field_config
    roles = fc.roles
    for required in ("target", "global"):
        if required not in roles:
            raise ValueError(
                f"field roles must define {required!r}; got {sorted(roles)}"
            )
    checks = [("target", "obs_vars"), ("global", "nwp_vars")]
    if "geo_bias" in roles:
        checks.append(("geo_bias", "geo_vars"))
    for role, scalar in checks:
        field_out = list(fc.fields[roles[role]].out)
        expected = [str(v) for v in cfg.dataset.get(scalar, [])]
        if field_out != expected:
            raise ValueError(
                f"role {role!r} field {roles[role]!r} channels {field_out} "
                f"!= cfg.dataset.{scalar} {expected} (order matters)"
            )


def setup_field_data_and_transform(cfg: DictConfig, logger: logging.Logger) -> tuple[
    Any,
    int,
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    T.NestedTransform,
    T.Transform,
]:
    """Field-path counterpart of ``setup.setup_data_and_transform``."""
    from oat.data_module.setup import _instantiate_dataset  # avoid import cycle

    dataset: Any = _instantiate_dataset(cfg)
    validate_field_config_against_scalars(dataset, cfg)
    logger.info(f"Dataset size: {len(dataset)} samples")

    splits = create_field_splits(dataset, **cfg.split, logger=logger)
    train_dataset, val_dataset, test_dataset = splits["datasets"]
    train_idx, val_idx, test_idx = splits["station_indices"]

    if all(
        cfg.split.get(k) is None
        for k in (
            "train_station_indices",
            "val_station_indices",
            "test_station_indices",
        )
    ):
        cfg.split.train_station_indices = train_idx
        cfg.split.val_station_indices = val_idx
        cfg.split.test_station_indices = test_idx

    # Stats over TRAIN stations, restricted to train/val time steps.
    inner_by_field, output_transform = dataset.compute_field_leaves(
        num_data_points=cfg.dataset.data_points_for_stats,
        data_indices=train_dataset._time_indices,
        target_point_indices=np.asarray(train_idx),
    )
    input_transform = build_two_group_transform(inner_by_field)

    train_loader = DataLoader(
        train_dataset, shuffle=True, collate_fn=field_collate, **cfg.dataloader.train
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=field_collate,
        **cfg.dataloader.validation,
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, collate_fn=field_collate, **cfg.dataloader.test
    )
    return (
        dataset,
        len(train_dataset),
        train_loader,
        val_loader,
        test_loader,
        input_transform,
        output_transform,
    )
